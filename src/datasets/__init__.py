"""数据集统一对外接口。

这个包原来是单个 ``datasets.py``（693 行），一个文件里塞了五件事：
读图、预处理管道、标签映射、Mini-ImageNet 数据集、条码数据集。
拆开后按职责分成五个模块：

    image_io.py       读图（中文路径兼容、通道统一）
    transforms.py     归一化常量、预处理管道、增强类型常量
    labels.py         标签映射构建
    mini_imagenet.py  Mini-ImageNet 数据集
    barcode.py        条码数据集

**调用方一行都不用改**：本文件把原来对外可见的名字原样重新导出，
所以 ``from datasets import BarcodeDataset`` 这类写法继续有效。
包内模块之间用相对导入（``from .labels import ...``），不依赖 src 是否
在 sys.path 上，也不怕以后包被移动到别的位置。
"""

from .barcode import BarcodeDataset
from .image_io import read_barcode_image, read_image_any_path
from .labels import build_barcode_class_mapping, build_label_to_index_map
from .mini_imagenet import Mini_ImageNet_Datasets
from .profile import ProfileDataset
from .transforms import (
    AUGMENT_C39,
    AUGMENT_NONE,
    AUGMENT_SYN,
    NORM_MEAN,
    NORM_STD,
    build_barcode_preprocess_pipeline,
    build_eval_transform,
)

__all__ = [
    "AUGMENT_C39",
    "AUGMENT_NONE",
    "AUGMENT_SYN",
    "BarcodeDataset",
    "Mini_ImageNet_Datasets",
    "NORM_MEAN",
    "NORM_STD",
    "ProfileDataset",
    "build_barcode_class_mapping",
    "build_barcode_preprocess_pipeline",
    "build_eval_transform",
    "build_label_to_index_map",
    "read_barcode_image",
    "read_image_any_path",
]
