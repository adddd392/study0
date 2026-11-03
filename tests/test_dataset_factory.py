"""dataset_factory 的验收测试：注册表 + 两套数据集的统一构建。

契约::

    bundle = build_split(dataset_name, split, cfg, shared_map=None)

    1. 【注册表】新增数据集只注册一行；未知名字必须报错。
    2. 【通道数由数据集声明】Mini-ImageNet 是 RGB(3)，条码集默认灰度(1)，
       并且这个值会传给 model_init 用于构建第一层卷积。
    3. 【映射一致】val/test 的类别映射必须等于训练集那一份，不一致就报错，
       而不是让类别下标静默错位。
    4. 【端到端】Trainer 按 dataset 名切换数据集，1 通道数据能喂进模型。

运行::

    python tests/test_dataset_factory.py
"""

import argparse
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402

from dataset_factory import (  # noqa: E402
    available_datasets,
    build_split,
)
from train import Trainer  # noqa: E402

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
MINI_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "data",
                                         "Mini-ImageNet-Dataset"))
BARCODE_DIR = os.path.join(PROJECT_ROOT, "dataset")

# 条码三份切分产物（由 src/split_barcode.py 生成并入库）
BARCODE_LENGTHS = {"train": 4569, "val": 979, "test": 979}
BARCODE_MAP = {"C128": 0, "EAN13": 1}


def barcode_args(**overrides):
    """构造条码任务的配置。"""
    base = {
        "dataset": "barcode",
        "device": "cpu",
        "input_size": (224, 224),
        "batch_size": 2,
        "num_workers": 0,
        "keep_grayscale": True,
        "model_name": "Resnet-50",
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "criterion": "CrossEntropyLoss",
        "save_path": os.path.join(PROJECT_ROOT, ".tmp_tests", "factory"),
        "use_train_csv": True,
        "train_csv": os.path.join(BARCODE_DIR, "barcode_train.csv"),
        "use_val_csv": True,
        "val_csv": os.path.join(BARCODE_DIR, "barcode_val.csv"),
        "use_test_csv": True,
        "test_csv": os.path.join(BARCODE_DIR, "barcode_test.csv"),
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def mini_args(**overrides):
    """构造 Mini-ImageNet 的配置。"""
    base = {
        "dataset": "mini_imagenet",
        "device": "cpu",
        "data_root": MINI_ROOT,
        "input_size": (64, 64),
        "batch_size": 4,
        "num_workers": 0,
        "use_train_csv": True,
        "train_csv": os.path.join(MINI_ROOT, "train.csv"),
        "use_val_csv": True,
        "val_csv": os.path.join(MINI_ROOT, "val.csv"),
        "use_test_csv": False,
        "test_csv": os.path.join(MINI_ROOT, "test.csv"),
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def case_registry() -> None:
    """两套数据集都已注册，未知名字报错。"""
    names = available_datasets()
    assert "mini_imagenet" in names, f"注册表缺 mini_imagenet: {names}"
    assert "barcode" in names, f"注册表缺 barcode: {names}"

    try:
        build_split("不存在的数据集", "train", barcode_args())
    except ValueError as exc:
        assert "未知数据集" in str(exc), f"报错信息不明确: {exc}"
    else:
        raise AssertionError("未知数据集应抛 ValueError")
    print("注册表: OK")


def case_mini_imagenet_bundle() -> None:
    """Mini-ImageNet 的 bundle：RGB 三通道，映射 100 类。"""
    if not os.path.isdir(MINI_ROOT):
        print("Mini-ImageNet: 跳过（数据不存在）")
        return
    bundle = build_split("mini_imagenet", "train", mini_args())
    assert bundle.in_channels == 3, f"应为 RGB 三通道，实际 {bundle.in_channels}"
    assert len(bundle.label_to_index_map) == 100, (
        f"类别数应为 100，实际 {len(bundle.label_to_index_map)}"
    )
    image, label = bundle.dataset[0]
    assert tuple(image.shape) == (3, 64, 64), f"形状错: {tuple(image.shape)}"
    assert isinstance(label, int), f"标签应为 int，实际 {type(label)}"
    print("Mini-ImageNet bundle: OK")


def case_barcode_bundle() -> None:
    """条码集 bundle：单通道、固定两类映射、三份长度正确。"""
    train = build_split("barcode", "train", barcode_args())
    assert train.in_channels == 1, (
        f"灰度数据集应声明 1 通道，实际 {train.in_channels}"
    )
    assert train.label_to_index_map == BARCODE_MAP, (
        f"类别映射错: {train.label_to_index_map}"
    )

    # 训练集自己产生映射，val/test 复用同一份
    for split in ("val", "test"):
        bundle = build_split("barcode", split, barcode_args(),
                             train.label_to_index_map)
        assert bundle.label_to_index_map == train.label_to_index_map, (
            f"{split} 的映射与训练集不一致"
        )
        assert len(bundle.dataset) == BARCODE_LENGTHS[split], (
            f"{split} 样本数期望 {BARCODE_LENGTHS[split]}，"
            f"实际 {len(bundle.dataset)}"
        )

    image, label = train.dataset[0]
    assert tuple(image.shape) == (1, 224, 224), f"形状错: {tuple(image.shape)}"
    assert image.dtype.is_floating_point, f"应为浮点张量，实际 {image.dtype}"
    assert 0 <= label < 2, f"标签越界: {label}"
    print("条码 bundle: OK")


def case_barcode_keep_grayscale_false() -> None:
    """keep_grayscale=False 时声明三通道，输出也确实是三通道。"""
    bundle = build_split(
        "barcode", "train", barcode_args(keep_grayscale=False)
    )
    assert bundle.in_channels == 3, (
        f"彩色模式下应声明 3 通道，实际 {bundle.in_channels}"
    )
    image, _ = bundle.dataset[0]
    assert tuple(image.shape) == (3, 224, 224), f"形状错: {tuple(image.shape)}"
    print("keep_grayscale=False 声明三通道: OK")


def case_mismatched_map_rejected() -> None:
    """val 的映射与训练集不一致时必须报错，不能静默错位。"""
    wrong_map = {"C128": 0, "EAN13": 1, "C39": 2}
    try:
        build_split("barcode", "val", barcode_args(), wrong_map)
    except ValueError as exc:
        assert "映射" in str(exc), f"报错信息不明确: {exc}"
    else:
        raise AssertionError("映射不一致时 val 应抛 ValueError")
    print("映射不一致被拒绝: OK")


def case_unknown_barcode_class_rejected() -> None:
    """fixed_classes 缺类时必须报错。"""
    try:
        build_split(
            "barcode", "train",
            barcode_args(fixed_classes=("C128",)),
        )
    except ValueError as exc:
        assert "未知类别" in str(exc), f"报错信息不明确: {exc}"
    else:
        raise AssertionError("缺类别时构建应抛 ValueError")
    print("缺类别被拒绝: OK")


def case_trainer_end_to_end() -> None:
    """Trainer 按数据集名切换：建数据 -> 定通道数 -> 建模型 -> 前向一批。"""
    args = barcode_args()
    obj = Trainer(args)

    num_classes = obj.dataset_init()
    assert num_classes == 2, f"类别数期望 2，实际 {num_classes}"
    assert obj.in_channels == 1, (
        f"in_channels 应跟着数据集走（1），实际 {obj.in_channels}"
    )

    # 模型第一层必须被换成 1 通道
    obj.model_init(num_classes)
    first_conv = obj.model.conv1
    assert first_conv.in_channels == 1, (
        f"模型第一层仍是 {first_conv.in_channels} 通道，喂 1 通道数据会报错"
    )

    # 真正取一批数据前向：这条链路通了，才说明数据集和模型都对齐了
    images, labels = next(iter(obj.train_loader))
    assert tuple(images.shape[1:]) == (1, 224, 224), (
        f"批张量形状错: {tuple(images.shape)}"
    )
    obj.model.eval()
    with torch.no_grad():
        logits = obj.model(images)
    assert tuple(logits.shape) == (2, 2), f"输出形状错: {tuple(logits.shape)}"
    print("Trainer 端到端（条码）: OK")


def case_worker_init_hook() -> None:
    """BarcodeDataset 的 worker_init_fn 应被挂到 DataLoader 上。"""
    obj = Trainer(barcode_args())
    obj.dataset_init()
    assert obj.train_loader.worker_init_fn is not None, (
        "条码 DataLoader 未挂 worker_init_fn，多进程下 cv2 会和 PyTorch 抢核"
    )
    print("worker_init_fn 已挂载: OK")


def main() -> int:
    case_registry()
    case_mini_imagenet_bundle()
    case_barcode_bundle()
    case_barcode_keep_grayscale_false()
    case_mismatched_map_rejected()
    case_unknown_barcode_class_rejected()
    case_trainer_end_to_end()
    case_worker_init_hook()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
