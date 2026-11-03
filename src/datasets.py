"""Mini-ImageNet /BarcodeDataset数据集与图像预处理管道。

训练与推理必须共用同一套预处理,否则会出现"训练时一个分布、推理时另一个分布"
的隐性错误。因此这里把归一化常量、预处理管道和标签映射统一抽成模块级函数,
供 train.py / inference.py 复用。
"""

import os
import time

import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    def __init__(self, csv_file, data_root, inputsize=(224, 224),
                 label_to_index_map=None):
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
        image_path = os.path.join(
            self.data_root, row["label"], row["filename"])
        image = plt.imread(image_path)
        image = self.transform(image=image)["image"]

        label = row["label"]
        label = self.label_to_index_map[label]
        return image, label

    def __len__(self):
        return len(self.df)


# ---------------------------------------------------------------------------
# 条码数据集:常量、映射构建与预处理管道
# ---------------------------------------------------------------------------

# 增强类型的标记值。不直接存 transform 对象,而是在初始化时把每个样本
# 归到下面三类之一,__getitem__ 里只需要一次整数比较,不必再读 DataFrame。
AUGMENT_NONE = 0   # 走通用 transform
AUGMENT_C39 = 1    # 走 transform_c39
AUGMENT_SYN = 2    # 走 transform_syn(优先级最高)


def build_barcode_preprocess_pipeline(target_size, keep_grayscale=True,
                                      interpolation=cv2.INTER_AREA):
    """构建条码数据集的基础预处理管道:Resize -> Normalize -> ToTensorV2。

    这条管道是「必经之路」:无论有没有数据增强,最后都由它把图像变成
    形状固定、取值固定的张量。因此增强管道不必各自再带 ToTensorV2。

    参数:
        target_size: (H, W) 输出尺寸。
        keep_grayscale: True 时按单通道归一化,否则按三通道。
        interpolation: 缩放插值方式。默认 INTER_AREA ——
            条码靠「条空宽窄比例」区分,缩小图像时 INTER_NEAREST 会丢掉
            整条黑边(实测 74 个条空边界只剩 53 个),INTER_AREA 是
            缩小场景的正确选择。
    返回:
        albumentations.Compose,调用 transform(image=ndarray)['image']。
    """
    # 归一化常量必须和通道数匹配:灰度 1 元组,彩色 3 元组
    if keep_grayscale:
        norm_mean, norm_std = (0.5,), (0.5,)
    else:
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    return A.Compose([
        # 统一尺寸:不 Resize 的话每张图尺寸都不同,DataLoader 无法拼 batch
        A.Resize(
            height=target_size[0],
            width=target_size[1],
            interpolation=interpolation,
            p=1.0,
        ),
        # 0~255 -> [-1, 1]
        A.Normalize(mean=norm_mean, std=norm_std),
        # HWC(float32) -> CHW(torch.Tensor)
        ToTensorV2(),
    ])


def build_barcode_class_mapping(label_names, fixed_classes=None):
    """构建 (class_names, class_to_idx),并拒绝 CSV 里的未知类别。

    为什么要有固定类别:训练与推理必须用同一套下标语义。若每次按 CSV
    内容排序生成,换一份 CSV 就可能让 C128 从 0 变成 1,预测结果整体错位。
    因此生产环境应始终传 fixed_classes。

    参数:
        label_names: CSV 中出现过的类别名。会先统一 strip + 大写,
            因此 'c128 ' 和 'C128' 视为同一类。
        fixed_classes: 固定类别次序(同样会归一化);None 表示按内容排序生成。
    返回:
        (class_names, class_to_idx),形如
        (['C128', 'EAN13'], {'C128': 0, 'EAN13': 1})
    异常:
        ValueError: CSV 中出现了 fixed_classes 之外的类别。
    """
    # 两侧都归一化,调用方就不必自己记得先转大写
    normalized = {str(name).strip().upper() for name in label_names}
    if fixed_classes is None:
        # 兜底路径:没有给定次序时才按内容排序
        class_names = sorted(normalized)
    else:
        # 统一转大写,避免 fixed_classes 里写 'c128' 就匹配不上
        class_names = [str(name).strip().upper() for name in fixed_classes]
        unknown = normalized - set(class_names)
        if unknown:
            # 必须报错而不是忽略:被忽略的样本会让标签变 NaN,
            # 进而让 CrossEntropyLoss 收到浮点标签并在训练中段才崩
            raise ValueError(
                f"CSV 包含未知类别: {sorted(unknown)},"
                f"当前 fixed_classes={class_names}。"
                "请把新类别加入 fixed_classes,或确认数据是否混入了别的码制。"
            )
    class_to_idx = {
        name: index
        for index, name in enumerate(class_names)
    }
    return class_names, class_to_idx


class BarcodeDataset(Dataset):
    """条码码制分类数据集:CSV -> (图像张量, 整数标签)。

    CSV 必需列:
        image_path  : 图像路径。绝对路径直接使用;相对路径配合 data_root。
        label_name  : 类别名(码制),内部统一转成大写再映射成整数。
    CSV 可选列:
        source_domain : 取值 'synthetic' 时该样本走 transform_syn。

    设计要点:
        1. 基础管道(Resize+Normalize+ToTensorV2)恒执行,且只在初始化时
           构建一次。因此输出形状永远等于 target_size,与是否提供 transform
           无关 —— 忘记传 transform 不会再得到尺寸各异的张量。
        2. 增强管道(transform / transform_c39 / transform_syn)只负责改像素,
           不要再包含 ToTensorV2;误包含会立刻抛 TypeError 并指明原因。
        3. 路径、标签、增强类型在初始化时全部预解析成数组,
           __getitem__ 里不再有任何 pandas 开销(旧实现每样本 2 次 iloc)。

    用法:
        dataset = BarcodeDataset(
            csv_file="dataset/barber.csv",
            target_size=(224, 224),
            fixed_classes=("C128", "EAN13"),
        )
        loader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=4,
            worker_init_fn=BarcodeDataset.worker_init_fn,
        )
    """

    # 增强失败的容忍上限:超过就直接报错。
    # 为什么要有上限:旧实现逐张静默降级,transform 写错时训练照跑、
    # loss 照降,但预处理分布已经变了,唯一的线索是几千行警告刷屏。
    MAX_TRANSFORM_FAILURES = 20

    def __init__(self, csv_file, transform=None, target_size=(224, 224),
                 data_root=None, keep_grayscale=True,
                 fixed_classes=("C128", "EAN13"), cache_images=False,
                 transform_c39=None, transform_syn=None,
                 interpolation=cv2.INTER_AREA, verbose=True):
        """初始化数据集。

        参数:
            csv_file: CSV 路径。
            transform: 通用数据增强管道(只改像素,不含 ToTensorV2)。
            target_size: (H, W) 输出尺寸,由基础管道负责实现。
            data_root: 图像根目录;CSV 里是相对路径时用它拼接。
            keep_grayscale: True 输出单通道,False 输出三通道 RGB。
            fixed_classes: 固定类别次序。默认值用 tuple 而不是 list ——
                list 作为默认参数会在所有实例间共享,一旦有人 append
                就会污染其它实例。
            cache_images: 是否把所有原图解码后常驻内存(省每 epoch 的解码)。
            transform_c39: C39 专属增强。仅当 fixed_classes 含 'C39'
                时才可能生效,否则 CSV 里的 C39 样本会先被判为未知类别。
            transform_syn: 合成数据专属增强,优先级高于 transform_c39。
            interpolation: 基础管道缩放插值方式,默认 INTER_AREA。
            verbose: 是否打印数据集概况。
        """
        self.csv_file = csv_file
        self.target_size = tuple(target_size)
        self.data_root = data_root
        self.keep_grayscale = keep_grayscale
        self.verbose = verbose

        # ---- 增强管道:只改像素,基础管道会在最后统一归一化 ----
        self.transform = transform
        self.transform_c39 = transform_c39
        self.transform_syn = transform_syn

        # ---- 读 CSV 并统一标签书写 ----
        self.df = pd.read_csv(csv_file)
        # strip 去掉首尾空白,upper 统一大小写:
        # 否则 'c128 ' 和 'C128' 会被当成两个不同的类
        self.df['label_name'] = (
            self.df['label_name'].astype(str).str.strip().str.upper()
        )

        # ---- 类别映射 ----
        self.class_names, self.class_to_idx = build_barcode_class_mapping(
            label_names=self.df['label_name'].unique(),
            fixed_classes=fixed_classes,
        )
        self.num_classes = len(self.class_names)
        # 把整数标签写回 df:外部可按类别过滤,也可用 df['label_id'] 统计
        self.df['label_id'] = (
            self.df['label_name'].map(self.class_to_idx).astype('int64')
        )

        # ---- 基础预处理管道 ----
        # 只建一次:旧实现每取一个样本都重新 new 一个 A.Compose,
        # 6527 张图乘每个 epoch 都是白白的对象分配开销
        self.base_pipeline = build_barcode_preprocess_pipeline(
            target_size=self.target_size,
            keep_grayscale=keep_grayscale,
            interpolation=interpolation,
        )

        # ---- 预解析:让 __getitem__ 彻底不碰 pandas ----
        self._image_paths = self._resolve_all_paths()
        self._labels = self.df['label_id'].to_numpy()
        self._augment_tags = self._resolve_augment_tags()

        # ---- 可选的图像缓存 ----
        self.cache_images = cache_images
        self._cache = None
        self._transform_failures = 0
        if verbose:
            self._print_summary()
        if cache_images:
            self._preload_images()

    # -- 初始化期的辅助方法 -------------------------------------------------

    def _resolve_all_paths(self):
        """把 CSV 的 image_path 一次性解析成可直接交给 cv2.imread 的列表。

        绝对路径原样返回;相对路径用 data_root 拼接(data_root 为 None 时
        保留原样,由 cv2 相对当前工作目录解析)。
        """
        paths = []
        for raw_path in self.df['image_path']:
            # CSV 读进来可能是 numpy.str_,且可能带首尾空格
            path = str(raw_path).strip()
            if self.data_root is not None and not os.path.isabs(path):
                path = os.path.join(str(self.data_root), path)
            paths.append(path)
        return paths

    def _resolve_augment_tags(self):
        """预先算出每个样本该用哪条增强管道,返回 int8 数组。

        优先级:合成数据(transform_syn) > C39(transform_c39) > 通用。
        未提供对应管道时,该类样本一律回落到 AUGMENT_NONE。
        """
        tags = np.full(len(self._image_paths), AUGMENT_NONE, dtype=np.int8)
        # 合成数据优先级最高,先打标
        if self.transform_syn is not None and (
            'source_domain' in self.df.columns
        ):
            # 统一成小写再比较,避免 'Synthetic' / ' synthetic ' 漏标
            domain = (
                self.df['source_domain'].astype(str).str.strip().str.lower()
            )
            tags[(domain == 'synthetic').to_numpy()] = AUGMENT_SYN
        # C39 只覆盖还没被标成合成数据的样本
        if self.transform_c39 is not None:
            is_c39 = (self.df['label_name'] == 'C39').to_numpy()
            tags[is_c39 & (tags == AUGMENT_NONE)] = AUGMENT_C39
        return tags

    def _print_summary(self):
        """打印数据集概况:类别、映射、各类样本数,便于训练前核对。"""
        counts = self.df['label_name'].value_counts()
        lines = [
            f"\n[BarcodeDataset] {len(self.df)} 张,{self.num_classes} 类",
            f"  CSV: {self.csv_file}",
            f"  class_names: {self.class_names}",
            f"  class_to_idx: {self.class_to_idx}",
            "  各类样本数:",
        ]
        for name in self.class_names:
            lines.append(f"    {name}: {int(counts.get(name, 0))}")
        print("\n".join(lines))

    def _preload_images(self):
        """把所有原图解码一次并常驻内存,省掉每个 epoch 的解码开销。

        注意:缓存的是「未缩放原图」,内存占用取决于原图尺寸。
        本数据集实测约 0.24 MB/张,6527 张约 1.6 GB,开启前请先估算。
        """
        start = time.time()
        if self.verbose:
            print(f"[BarcodeDataset] 预加载 {len(self._image_paths)} 张...")
        # 顺序必须与 _image_paths 一致,__getitem__ 才能直接用下标命中
        self._cache = [self._read_image(p) for p in self._image_paths]
        total_gb = sum(image.nbytes for image in self._cache) / 1e9
        if self.verbose:
            print(f"[BarcodeDataset] 缓存完成: {total_gb:.2f} GB,"
                  f" 耗时 {time.time() - start:.0f}s")

    # -- 单样本读取 ---------------------------------------------------------

    @staticmethod
    def _imread_any_path(img_path):
        """读图,兼容含中文/空格的路径。

        为什么不用 cv2.imread:它在 Windows 上走窄字符 API,路径里只要
        出现中文就静默返回 None(本项目仓库路径就带中文,实测必踩)。
        改用「numpy 读字节 + cv2.imdecode」,把路径解析交给 Python 的文件层,
        OpenCV 只负责解码内存中的字节,不再碰路径。

        异常:
            FileNotFoundError: 文件不存在、为空或无法解码。
        """
        try:
            buffer = np.fromfile(img_path, dtype=np.uint8)
        except OSError as exc:
            # 文件不存在、是目录、无权限都归到这里
            raise FileNotFoundError(f"图像无法读取: {img_path}") from exc
        if buffer.size == 0:
            raise FileNotFoundError(f"图像为空文件: {img_path}")
        # IMREAD_UNCHANGED 保留原始通道数,由调用方决定怎么转换
        img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"图像无法解码(格式不支持?): {img_path}")
        return img

    def _read_image(self, img_path):
        """读一张图,并按 keep_grayscale 统一通道数。

        返回:
            keep_grayscale=True  -> (H, W) 单通道 ndarray
            keep_grayscale=False -> (H, W, 3) RGB ndarray
        异常:
            FileNotFoundError: 路径不存在或文件无法解码。
        """
        img = self._imread_any_path(img_path)
        channels = 1 if img.ndim == 2 else img.shape[2]
        if self.keep_grayscale:
            # 彩色/带 alpha 的一律转灰度;本来就是灰度的原样返回
            if channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            # 统一成 RGB 三通道:OpenCV 默认是 BGR,不转换会让红蓝互换
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(f"不支持的图像通道数: {channels} ({img_path})")
        return img

    def _select_augment(self, idx):
        """返回该样本应使用的增强管道;没有则为 None。"""
        tag = int(self._augment_tags[idx])
        if tag == AUGMENT_SYN:
            return self.transform_syn
        if tag == AUGMENT_C39:
            return self.transform_c39
        return self.transform

    def _apply_augment(self, img, idx):
        """对图像应用该样本选中的增强管道,返回值仍是 ndarray。

        异常处理策略:
            - 前 MAX_TRANSFORM_FAILURES 次:降级为基础预处理,只打一条警告;
            - 超过上限:抛 RuntimeError,让错误在第一批 batch 内暴露,
              而不是让训练带着错误的预处理跑完整个 epoch。
        """
        augment = self._select_augment(idx)
        if augment is None:
            # 没有增强,直接交给基础管道
            return img
        try:
            augmented = augment(image=img)['image']
        except Exception as exc:
            # 捕获面故意放宽:第三方增强库的异常类型不可控,
            # 但用「次数上限 + 首条警告」防止它变成静默降级
            self._handle_augment_failure(exc, idx)
            return img
        # 增强后必须还是 ndarray;拿到张量说明调用方把 ToTensorV2 也塞进来了。
        # 在 try 之外抛,避免被上面的 except 吞掉
        if isinstance(augmented, torch.Tensor):
            raise TypeError(
                "transform / transform_c39 / transform_syn 不应包含 "
                "ToTensorV2:数据集会在增强之后统一执行 "
                "Resize -> Normalize -> ToTensorV2。"
                f" 出错样本: {self._image_paths[idx]}"
            )
        return augmented

    def _handle_augment_failure(self, exc, idx):
        """统计并处理增强失败:超限报错,否则只报一次警。"""
        self._transform_failures += 1
        if self._transform_failures > self.MAX_TRANSFORM_FAILURES:
            raise RuntimeError(
                f"增强管道失败超过 {self.MAX_TRANSFORM_FAILURES} 次,"
                "已停止以避免用错误的预处理继续训练。"
                f"最近一次样本: {self._image_paths[idx]}"
            ) from exc
        if self._transform_failures == 1:
            print(
                f"[BarcodeDataset] 增强失败,已降级为基础预处理:"
                f"{self._image_paths[idx]} ({exc})"
            )
        elif self._transform_failures == self.MAX_TRANSFORM_FAILURES:
            print(
                f"[BarcodeDataset] 增强失败已达 "
                f"{self.MAX_TRANSFORM_FAILURES} 次,再失败将直接报错"
            )

    # -- Dataset 接口 -------------------------------------------------------

    def __len__(self):
        """样本总数。"""
        return len(self._image_paths)

    def __getitem__(self, idx):
        """返回 (image, label)。

        image: float32 张量,(1, H, W) 或 (3, H, W),H/W 等于 target_size。
        label: python int,取值 [0, num_classes)。
        """
        # pandas 支持负下标,但下面的 list 缓存不支持,先统一成正索引
        if idx < 0:
            idx += len(self._image_paths)
        img_path = self._image_paths[idx]

        # 命中缓存就不必再解码,这是 cache_images 唯一的意义
        if self.cache_images:
            img = self._cache[idx]
        else:
            img = self._read_image(img_path)

        # 1) 该样本专属的增强(可能为无)
        img = self._apply_augment(img, idx)

        # 2) 基础管道:恒执行,所以输出形状只由 target_size 决定
        tensor = self.base_pipeline(image=img)['image']

        # 3) 标签从预解析的数组取,避免每样本一次 pandas 索引
        return tensor, int(self._labels[idx])

    # -- 工具方法 -----------------------------------------------------------

    def validate_files(self, limit=None):
        """检查图像是否都在磁盘上,返回缺失/无法读取的路径列表。

        CSV 里存的是绝对路径,数据目录一旦搬家,错误要等 __getitem__
        被调用才暴露。训练前跑一次本方法可以提前发现,避免跑到一半崩。
        """
        paths = self._image_paths if limit is None else (
            self._image_paths[:limit]
        )
        return [path for path in paths if not os.path.isfile(path)]

    def get_map(self):
        """返回 class_to_idx,便于与推理端的映射对齐。"""
        return self.class_to_idx

    @staticmethod
    def worker_init_fn(worker_id):
        """DataLoader 的 worker_init_fn:关掉 cv2 自带的线程池。

        num_workers > 0 时,每个 worker 会自己再开一套 OpenCV 线程,
        和 PyTorch 抢 CPU 核,典型表现是「加了 worker 反而更慢」。
        用法:DataLoader(..., worker_init_fn=BarcodeDataset.worker_init_fn)
        """
        cv2.setNumThreads(0)

    def __repr__(self):
        """调试用的简短描述,不打印逐类统计。"""
        return (
            f"BarcodeDataset(samples={len(self)},"
            f" classes={self.num_classes},"
            f" target_size={self.target_size},"
            f" keep_grayscale={self.keep_grayscale})"
        )


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
        target_size=(224, 224),
        # 不需要再传 transform:Resize -> Normalize -> ToTensorV2 这条基础
        # 管道在数据集内部恒执行,输出形状只由 target_size 决定
    )
    # 训练前先确认图像都在盘上:CSV 存的是绝对路径,目录一搬就会抛错
    missing = barcode_dataset.validate_files()
    print(f"[冒烟] 缺失图像 = {len(missing)} 个")

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

    # ------------------------------------------------------------------
    # 样本可视化:每个类别各取 2 张,排成一行
    # 断言只能证明"形状和数值范围对",证明不了"标签和图像对得上";
    # 把图和类别名摆在一起,是唯一能一眼看出来的检查方式
    # ------------------------------------------------------------------
    # 中文字体:matplotlib 默认字体没有汉字,不加这行标题会显示成方块
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    # 负号用 ASCII 显示,避免部分字体把负号渲染成缺字方块
    plt.rcParams['axes.unicode_minus'] = False

    # 逐类别挑前 2 个下标:按类别取样,才能同时看到两种码制
    show_indices = []
    for class_name in barcode_dataset.class_names:
        # 该类别在 DataFrame 中的全部行号(行号可直接喂给 __getitem__)
        rows = barcode_dataset.df.index[
            barcode_dataset.df['label_name'] == class_name
        ].tolist()
        show_indices.extend(rows[:2])

    # 列数随实际取到的张数变化,避免某一类样本不足时子图空着
    n_show = len(show_indices)
    fig, axes = plt.subplots(1, n_show, figsize=(3.2 * n_show, 3.4))
    # n_show == 1 时 axes 是单个对象而不是数组,统一成可迭代
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, show_indices):
        vis_image, vis_label = barcode_dataset[idx]
        # (1, H, W) 且归一化到 [-1, 1] -> (H, W) 且 [0, 1] 才能直接显示:
        # squeeze 去掉通道维, x * 0.5 + 0.5 是 (x / 255 - 0.5) / 0.5 的逆变换
        vis_image = vis_image.squeeze(0).numpy() * 0.5 + 0.5
        # 不写死 vmin/vmax:让 matplotlib 按每张图的实际范围拉伸对比度。
        # 条码照片普遍偏暗(归一化后常落在 0.0x~0.4),写死 0~1 会糊成一片;
        # 真实数值范围由上面对 min/max 的断言负责,不靠这张图判断
        ax.imshow(vis_image, cmap='gray')
        # 标题写成「行号 + 类别名」:行号可用于回到 CSV 逐行核对
        ax.set_title(
            f"#{idx}  {barcode_dataset.class_names[vis_label]}",
            fontsize=10,
        )
        # 关掉坐标轴:条码图的像素坐标没有信息量,留着只占地方
        ax.axis('off')
    fig.suptitle(f"BarcodeDataset 样本预览(共 {n_show} 张)", fontsize=12)
    fig.tight_layout()

    # 放到最后:先让测试输出打印完,再弹窗显示 Mini-ImageNet 的样本,
    # 否则 plt.show() 会阻塞,看不到后面的冒烟结果
    plt.show()
