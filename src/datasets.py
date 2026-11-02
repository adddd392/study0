"""Mini-ImageNet /BarcodeDataset数据集与图像预处理管道。

训练与推理必须共用同一套预处理,否则会出现"训练时一个分布、推理时另一个分布"
的隐性错误。因此这里把归一化常量、预处理管道和标签映射统一抽成模块级函数,
供 train.py / inference.py 复用。
"""

import os

import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# 训练与推理共用的归一化参数。

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def build_eval_transform(inputsize=(224, 224)):
    """构建确定性的评估/推理预处理管道:Resize -> Normalize -> ToTensorV2。

    参数:
        inputsize: (H, W) 目标尺寸。
    返回:
        albumentations.Compose 对象,调用方式为 transform(image=ndarray)['image']。
    """
    return A.Compose(
        [
            # 直接缩放到网络输入尺寸(本实验不含随机裁剪)
            A.Resize(height=inputsize[0], width=inputsize[1]),
            # 归一化到 [-1, 1]
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            # HWC(float32, 0~1) -> CHW(torch.Tensor)
            ToTensorV2(),
        ]
    )


def build_label_to_index_map(csv_file):
    """按训练时的规则构建 label -> index 映射。

    规则:按 label 在 CSV 中**首次出现的顺序**依次编号 0,1,2...

    推理时必须用 train.csv(而不是 val/test.csv)调用本函数,才能得到
    与训练完全一致的下标语义;否则预测出的下标对应的类别会整体错位。

    参数:
        csv_file: 训练集 CSV 路径,需包含 label 列。
    返回:
        dict,形如 {'n01532829': 0, 'n01558993': 1, ...}
    """
    df = pd.read_csv(csv_file)
    label_classes = df["label"].unique()
    return {label: i for i, label in enumerate(label_classes)}


class Mini_ImageNet_Datasets(Dataset):
    def __init__(self, csv_file, data_root, inputsize=(224, 224), label_to_index_map=None):
        self.df = pd.read_csv(csv_file)
        self.data_root = data_root
        self.inputsize = inputsize
        # 复用预先构建好的映射(验证集/测试集使用训练集的映射,保证下标语义一致)
        if label_to_index_map is None:
            self.label_to_index_map = build_label_to_index_map(csv_file)
        else:
            self.label_to_index_map = label_to_index_map
        # 预处理管道只构建一次,避免每个样本重复构造(原实现在 __getitem__ 中逐样本构建)
        self.transform = build_eval_transform(inputsize)

    def get_map(self):
        return self.label_to_index_map

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.data_root, row["label"], row["filename"])
        image = plt.imread(image_path)
        image = self.transform(image=image)["image"]

        label = row["label"]
        label = self.label_to_index_map[label]
        return image, label

    def __len__(self):
        return len(self.df)


class BarcodeDataset(Dataset):
    """
    条码数据集类，支持多种码制分类。
    自动从 CSV 中读取类别名称，分配整数标签。
    支持绝对路径或相对路径（需指定 data_root）。
    """
    def __init__(self, csv_file, transform=None, target_size=(224, 224), data_root=None, keep_grayscale=True,fixed_classes=['C128', 'EAN13'], cache_images=False, transform_c39=None, transform_syn=None):
        """
        Args:
            csv_file (str): labels.csv 文件路径。
            transform (albumentations.Compose, optional): 数据增强流水线。
            target_size (tuple): 输出图像尺寸 (height, width)。
            data_root (str, optional): 图像根目录，当 CSV 中为相对路径时拼接该路径。
            cache_images (bool): 是否预加载全部图像到内存缓存 (省每 epoch 的 imread IO/解码)。
            transform_c39 (albumentations.Compose, optional): C39 专属增强流水线。
                当样本的 label_name == 'C39' 时使用该流水线；为 None 时回退到 transform。
            transform_syn (albumentations.Compose, optional): 合成数据专属增强流水线。
                当样本的 source_domain == 'synthetic' 时使用该流水线（优先级最高）；
                为 None 时回退到 transform_c39 / transform。
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.transform_c39 = transform_c39
        self.transform_syn = transform_syn
        self.target_size = target_size
        self.data_root = data_root  # 新增
        self.keep_grayscale = keep_grayscale

        # ----- 关键修复：先将标签转为大写 -----
        self.df['label_name'] = self.df['label_name'].str.upper()

        # ----- 处理固定类别 -----
        if fixed_classes is not None:
            # 强制使用固定类别列表（也转为大写）
            self.class_names = [name.upper() for name in fixed_classes]
            # 检查 CSV 中是否包含未知类别
            unknown = set(self.df['label_name'].unique()) - set(self.class_names)
            if unknown:
                raise ValueError(f"CSV 包含未知类别: {unknown}，请调整 fixed_classes 或数据。")
        else:
            # 原逻辑：从 CSV 提取
            self.class_names = sorted(self.df['label_name'].unique())

        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # 添加整数标签列
        self.df['label_id'] = self.df['label_name'].map(self.class_to_idx)

        # ----- 添加调试输出 -----
        print(f"\n[BarcodeDataset] 初始化完成")
        print(f"  CSV 文件: {csv_file}")
        print(f"  类别名称 (class_names): {self.class_names}")
        print(f"  类别映射 (class_to_idx): {self.class_to_idx}")
        print(f"  每个类别的样本数:\n{self.df['label_name'].value_counts()}")
        print(f"  总样本数: {len(self.df)}\n")

        # ----- 可选: 预加载图像缓存 -----
        self.cache_images = cache_images
        self._cache = None
        if cache_images:
            import time as _t
            _t0 = _t.time()
            print(f"[BarcodeDataset] 预加载图像缓存 ({len(self.df)} 张)...", flush=True)
            self._cache = [self._read_image(self._resolve_path(i)) for i in range(len(self.df))]
            _mem = sum(g.nbytes for g in self._cache) / 1e9
            print(f"[BarcodeDataset] 缓存完成: 内存 {_mem:.2f} GB, 耗时 {_t.time()-_t0:.0f}s", flush=True)

    def _resolve_path(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        if self.data_root is not None and not os.path.isabs(img_path):
            img_path = os.path.join(self.data_root, img_path)
        return img_path

    def _read_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"图像不存在: {img_path}")
        if self.keep_grayscale:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(f"不支持的图像通道数: {img.shape}")
        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self._resolve_path(idx)
        label = self.df.iloc[idx]['label_id']

        # 读取图像（缓存或磁盘）
        if self.cache_images:
            img = self._cache[idx]
        else:
            img = self._read_image(img_path)

        # 若图像小于目标尺寸，先放大（最近邻）
        #h, w = img.shape[:2]
        #if h < self.target_size[0] or w < self.target_size[1]:
        #    img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 应用增强（如果有）—— 优先级: transform_syn(合成) > transform_c39(C39) > transform
        apply_transform = self.transform
        row = self.df.iloc[idx]
        if self.transform_syn is not None:
            src = str(row.get('source_domain', '')).strip().lower()
            if src == 'synthetic':
                apply_transform = self.transform_syn
        if apply_transform is self.transform and self.transform_c39 is not None:
            label_name = str(row['label_name']).strip().upper()
            if label_name == 'C39':
                apply_transform = self.transform_c39
        if apply_transform:
            try:
                augmented = apply_transform(image=img)
                img = augmented['image']
            except Exception as e:
                print(f"警告: 增强失败 ({e})，将使用默认预处理。图像: {img_path}")
                # 注意：默认归一化需要匹配通道数
                if self.keep_grayscale:
                    norm_mean = (0.5,)
                    norm_std = (0.5,)
                else:
                    norm_mean = (0.5, 0.5, 0.5)
                    norm_std = (0.5, 0.5, 0.5)
                default_transform = A.Compose([
                    A.Resize(height=self.target_size[0], width=self.target_size[1],
                            interpolation=cv2.INTER_NEAREST, p=1.0),
                    A.Normalize(mean=norm_mean, std=norm_std),
                    ToTensorV2()
                ])
                augmented = default_transform(image=img)
                img = augmented['image']
        else:
            # 无增强时的默认处理
            if self.keep_grayscale:
                norm_mean = (0.5,)
                norm_std = (0.5,)
            else:
                norm_mean = (0.5, 0.5, 0.5)
                norm_std = (0.5, 0.5, 0.5)
            default_transform = A.Compose([
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2()
            ])
            augmented = default_transform(image=img)
            img = augmented['image']

        return img, label

if __name__ == "__main__":
    dataset = Mini_ImageNet_Datasets(
        csv_file=r"..\data\Mini-ImageNet-Dataset\train.csv",
        data_root=r"..\data\Mini-ImageNet-Dataset\train",
    )
    print(len(dataset))
    image, label = dataset[0]
    image = image.permute(1, 2, 0).numpy()
    image = (image * 0.5 + 0.5) * 255
    image = image.astype(np.uint8)
    plt.imshow(image)

    # ------------------------------------------------------------------
    # 冒烟测试:新数据集 BarcodeDataset(barber.csv, C128 / EAN13 两类)
    # 目的不是评估精度,而是确认「CSV -> 路径解析 -> 读图 -> 预处理 ->
    # (张量, 标签)」这条链路能走通,且标签映射没有错位。
    # ------------------------------------------------------------------

    # 用绝对路径:脚本里的相对路径依赖启动目录,换目录运行就会读不到
    barcode_csv = r"D:\Users\admin\Desktop\代码\study\dataset\barber.csv"

    barcode_dataset = BarcodeDataset(
        csv_file=barcode_csv,
        # 复用模块级预处理管道,保证与训练/推理时同一套口径
        # (Resize -> Normalize -> ToTensorV2),不会出现分布漂移
        transform=build_eval_transform(inputsize=(224, 224)),
    )

    # 样本数必须等于 CSV 的行数,否则说明 CSV 解析或列名有问题
    print(f"[冒烟] 样本数 = {len(barcode_dataset)}")

    # 类别数与映射是标签语义的根:训练和推理必须用同一个映射,
    # 这里打印出来,方便和 train.py 里设置的 num_classes 对照
    print(f"[冒烟] 类别数 = {barcode_dataset.num_classes}")
    print(f"[冒烟] 类别映射 = {barcode_dataset.class_to_idx}")

    # 打印各类样本数:若某一类为 0,说明 CSV 里该类样本被漏读
    print(f"[冒烟] 类别分布 = "
          f"{barcode_dataset.df['label_name'].value_counts().to_dict()}")

    # 取第 0 个样本,跑通「读图 + 预处理 + 取标签」整条链路
    sample_image, sample_label = barcode_dataset[0]
    print(f"[冒烟] 样本0 shape = {tuple(sample_image.shape)}")
    print(f"[冒烟] 样本0 dtype = {sample_image.dtype}")
    print(f"[冒烟] 样本0 label = {sample_label}")

    # keep_grayscale 默认 True,所以是单通道:(C, H, W) = (1, 224, 224)。
    # 若之后要喂 3 通道主干(如 ResNet),需要传 keep_grayscale=False,
    # 期望值随之变成 (3, 224, 224)
    assert tuple(sample_image.shape) == (1, 224, 224), (
        f"期望 (1, 224, 224),实际 {tuple(sample_image.shape)}"
    )

    # 预处理后必须是浮点张量(Normalize + ToTensorV2 的结果)
    assert sample_image.dtype.is_floating_point, (
        f"期望浮点张量,实际 {sample_image.dtype}"
    )

    # Normalize 用 mean=std=0.5 把 0~255 映射到 -1~1,越界说明常量配错。
    # eps 只用来吸收浮点误差,不放宽语义
    eps = 1e-5
    assert float(sample_image.min()) >= -1.0 - eps, "归一化下界越界"
    assert float(sample_image.max()) <= 1.0 + eps, "归一化上界越界"

    # 标签必须落在 [0, num_classes),且能反查出类别名:
    # 这条不成立的话,训练能跑但每个 batch 的监督信号都是错的
    assert 0 <= sample_label < barcode_dataset.num_classes, (
        f"标签越界:{sample_label}"
    )
    print(f"[冒烟] 标签反查 = {barcode_dataset.class_names[sample_label]}")

    print("[冒烟] 通过:barber.csv 导入正常")

    # 放到最后:先让测试输出打印完,再弹窗显示 Mini-ImageNet 的样本,
    # 否则 plt.show() 会阻塞,看不到后面的冒烟结果
    plt.show()
