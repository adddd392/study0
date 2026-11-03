"""预处理管道与增强类型常量。

两套管道：
- build_eval_transform：Mini-ImageNet 用，固定三通道 RGB
- build_barcode_preprocess_pipeline：条码用，通道数可配（灰度 1 / 彩色 3）

共同点是「基础管道恒执行」：无论有没有数据增强，最后都由它把图像
变成形状固定、取值固定的张量。因此增强管道不必各自再带 ToTensorV2。
"""


import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
