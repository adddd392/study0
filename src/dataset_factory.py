"""数据集构建的统一入口:注册表 + builder。

为什么需要这一层
----------------
项目里有两套形态完全不同的数据集:

    Mini_ImageNet_Datasets      CSV 是 filename+label，路径拼 data_root/<label>/，
                                用 plt.imread，RGB 三通道，映射按 CSV 首次出现顺序
    BarcodeDataset              CSV 是 image_path+label_name，路径直接来自 CSV，
                                用 cv2 解码，灰度单通道，映射由 fixed_classes 决定

它们的构造参数、读图库、通道数、映射来源都不一样。如果在 Trainer 里写
if/else 分支，每加一个数据集就多一堆分支；如果硬抽一个公共基类，又只能
抽出没有任何信息的 __len__/__getitem__，反而把差异藏起来。

所以只统一"怎么建一份 split"这件事：每个数据集实现一个 builder 函数，
统一签名、统一返回 SplitBundle，Trainer 只查表调用。新增数据集 =
新增一个 builder + 注册一行，Trainer 一行都不用改。

两个数据集仍然只共享一个**鸭子协议**（写在文档里，不用继承）:

    dataset[i]            -> (tensor, int)
    dataset.get_map()     -> {类别名: 整数下标}
"""

from typing import Dict, NamedTuple

import os

from torch.utils.data import Dataset

from datasets import (
    BarcodeDataset,
    Mini_ImageNet_Datasets,
    build_barcode_class_mapping,
    build_barcode_preprocess_pipeline,
    build_eval_transform,
    build_label_to_index_map,
    read_barcode_image,
)


class SplitBundle(NamedTuple):
    """一份 split 的构建产物。

    字段:
        dataset: 可直接交给 DataLoader 的数据集。
        label_to_index_map: 这份 split 使用的类别映射。训练集由它自己产生，
            验证/测试集必须复用训练集那一份，否则类别下标整体错位。
        in_channels: 该数据集输出的通道数。**由数据集声明而不是写在配置里**，
            这样模型和数据集共用同一个真相来源，配置写错也不会出现
            "单通道数据喂三通道模型"。
    """

    dataset: Dataset
    label_to_index_map: Dict[str, int]
    in_channels: int


def _csv_for_split(cfg, split):
    """从配置里取某个 split 的 CSV 路径。"""
    csv_file = getattr(cfg, f"{split}_csv", None)
    if not csv_file:
        raise ValueError(f"配置缺少 {split} 的 CSV: cfg.{split}_csv")
    return csv_file


def _check_shared_map(dataset, shared_map, split):
    """校验验证/测试集的类别映射与训练集完全一致。

    这是全项目最关键的不变量:两套映射只要顺序不同，预测下标对应的类别
    就会整体错位，而且训练过程不会有任何报错。
    """
    if shared_map is None:
        return
    own_map = dataset.get_map()
    if own_map != shared_map:
        raise ValueError(
            f"{split} 的类别映射与训练集不一致。\n"
            f"  训练集: {shared_map}\n"
            f"  {split}: {own_map}\n"
            "请检查 fixed_classes / CSV 是否混入了训练集没有的类别。"
        )


def build_mini_imagenet_split(split, cfg, shared_map=None):
    """构建 Mini-ImageNet 的一份 split。

    路径约定: <data_root>/<split>/<label>/<filename>，所以 data_root 必须
    按 split 拼一段。
    """
    data_root = os.path.join(cfg.data_root, split)
    dataset = Mini_ImageNet_Datasets(
        csv_file=_csv_for_split(cfg, split),
        data_root=data_root,
        inputsize=cfg.input_size,
        label_to_index_map=shared_map,
    )
    return SplitBundle(
        dataset=dataset,
        label_to_index_map=dataset.get_map(),
        in_channels=3,
    )


def build_barcode_split(split, cfg, shared_map=None):
    """构建条码码制分类的一份 split。

    与 Mini-ImageNet 的三点差异:
        1. 路径直接来自 CSV（绝对路径），不需要按 split 拼 data_root；
        2. 图像是灰度，输出单通道，因此 in_channels=1；
        3. 类别映射由 fixed_classes 决定，不来自 CSV 内容 —— 只有这样
           才能保证训练/验证/测试的下标语义永远一致。
    """
    fixed_classes = _barcode_fixed_classes(cfg)
    keep_grayscale = getattr(cfg, "keep_grayscale", True)

    dataset = BarcodeDataset(
        csv_file=_csv_for_split(cfg, split),
        target_size=cfg.input_size,
        keep_grayscale=keep_grayscale,
        fixed_classes=fixed_classes,
        cache_images=getattr(cfg, "cache_images", False),
        # 三份 split 只在训练集上打印概况，否则日志会重复三遍
        verbose=(split == "train"),
    )
    _check_shared_map(dataset, shared_map, split)
    return SplitBundle(
        dataset=dataset,
        label_to_index_map=dataset.get_map(),
        in_channels=1 if keep_grayscale else 3,
    )


# 数据集名 -> builder。新增数据集只在这里加一行。
DATASET_REGISTRY = {
    "mini_imagenet": build_mini_imagenet_split,
    "barcode": build_barcode_split,
}


def available_datasets() -> list:
    """返回已注册的数据集名列表。"""
    return list(DATASET_REGISTRY.keys())


def build_split(dataset_name, split, cfg, shared_map=None) -> SplitBundle:
    """构建一份 split，是 Trainer 唯一需要调用的入口。

    参数:
        dataset_name: DATASET_REGISTRY 的键。
        split: "train" / "val" / "test"。
        cfg: 配置对象（argparse.Namespace），需含 input_size 与各 split 的 CSV。
        shared_map: 训练集的类别映射。训练集传 None，构建完由调用方取回。
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"未知数据集: {dataset_name},可选：{available_datasets()}"
        )
    return DATASET_REGISTRY[dataset_name](split, cfg, shared_map)


# ---------------------------------------------------------------------------
# 推理端:预处理与标签映射也必须按数据集分派
# ---------------------------------------------------------------------------

def _barcode_fixed_classes(cfg):
    """从配置读固定类别次序，统一大写。"""
    fixed_classes = getattr(cfg, "fixed_classes", ("C128", "EAN13"))
    return tuple(str(name).upper() for name in fixed_classes)


def build_label_map(dataset_name, cfg):
    """重建推理用的 类别名 -> 下标 映射。

    必须与训练时**同一个来源**，否则预测下标对应的类别整体错位，而且
    整个过程不会有任何报错:
        Mini-ImageNet  按 train.csv 里标签首次出现的顺序
        条码集          按 fixed_classes 常量的顺序
    """
    if dataset_name == "barcode":
        # 传空的 label_names:条码集的类别完全由 fixed_classes 决定，
        # 不依赖某个 CSV 恰好包含全部类别
        _, class_to_idx = build_barcode_class_mapping(
            (), _barcode_fixed_classes(cfg)
        )
        return class_to_idx
    if dataset_name == "mini_imagenet":
        return build_label_to_index_map(cfg.train_csv)
    raise ValueError(
        f"未知数据集: {dataset_name},可选：{available_datasets()}"
    )


def build_eval_preprocessor(dataset_name, cfg):
    """返回单张图片的推理预处理函数:: image_path -> (C, H, W) 张量。

    为什么必须按数据集分派:两套预处理读图库和通道处理都不同。条码图是
    灰度 PNG，若用 Mini-ImageNet 的 plt.imread + build_eval_transform
    走一遍，得到的张量与训练时的完全不是一回事，预测结果无法解释。
    """
    if dataset_name == "barcode":
        keep_grayscale = getattr(cfg, "keep_grayscale", True)
        pipeline = build_barcode_preprocess_pipeline(
            tuple(cfg.input_size), keep_grayscale=keep_grayscale
        )

        def preprocess_barcode(image_path):
            image = read_barcode_image(image_path, keep_grayscale)
            return pipeline(image=image)["image"]

        return preprocess_barcode

    if dataset_name == "mini_imagenet":
        # 延迟导入 pyplot：训练流程不需要它，避免白白拉高启动耗时
        import matplotlib.pyplot as plt

        transform = build_eval_transform(tuple(cfg.input_size))

        def preprocess_mini_imagenet(image_path):
            return transform(image=plt.imread(image_path))["image"]

        return preprocess_mini_imagenet

    raise ValueError(
        f"未知数据集: {dataset_name},可选：{available_datasets()}"
    )
