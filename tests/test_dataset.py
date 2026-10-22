"""trainer.dataset_init 的验收测试。

契约::

    num_classes = trainer(args).dataset_init()

    行为:
    - 按 args.use_train_csv / use_val_csv / use_test_csv 决定建哪几份数据；
    - 建好后挂在 self.train_dataset / self.train_loader（val、test 同理）；
    - train 的 DataLoader 打乱顺序，val/test 不打乱；
    - 【核心不变量】三份数据必须共用同一个 label_to_index_map，
      且该映射来自 train.csv（datasets.py 文档：val/test 若自行建映射，
      下标语义会与训练错位）；
    - 返回类别数，即 label_to_index_map 的长度。

    错误:
    - 未加载训练集却加载了 val/test -> ValueError
      （当前实现会让 val 自行用 val.csv 建映射，类别顺序与训练不同，
       下标整体错位且不报错；这里要求改为明确报错）

运行::

    python tests/test_dataset.py
"""

import argparse
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from datasets import build_label_to_index_map  # noqa: E402
from train import trainer  # noqa: E402

DATA_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "Mini-ImageNet-Dataset"
    )
)
EXPECTED_CLASSES = 100


def make_args(**overrides):
    """构造 dataset_init 需要的参数，默认三份数据全开。"""
    base = {
        "device": "cpu",
        "data_root": DATA_ROOT,
        "use_train_csv": True,
        "train_csv": os.path.join(DATA_ROOT, "train.csv"),
        "use_val_csv": True,
        "val_csv": os.path.join(DATA_ROOT, "val.csv"),
        "use_test_csv": True,
        "test_csv": os.path.join(DATA_ROOT, "test.csv"),
        "batch_size": 32,
        "input_size": (64, 64),
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def sampler_name(loader):
    """返回 loader 使用的采样器类名，用于判断是否打乱。"""
    return type(loader.batch_sampler.sampler).__name__


def case_all_splits() -> None:
    """三份数据全开：检查挂载、打乱策略、类别数与映射共享。"""
    args = make_args()
    obj = trainer(args)

    num_classes = obj.dataset_init()

    assert num_classes == EXPECTED_CLASSES, (
        f"类别数期望 {EXPECTED_CLASSES}，实际 {num_classes}"
    )

    # 三个 dataset / loader 都要挂上
    for split in ("train", "val", "test"):
        assert hasattr(obj, f"{split}_dataset"), f"缺少 {split}_dataset"
        assert hasattr(obj, f"{split}_loader"), f"缺少 {split}_loader"

    # 只有 train 打乱
    assert sampler_name(obj.train_loader) == "RandomSampler", (
        f"train 应打乱，实际 {sampler_name(obj.train_loader)}"
    )
    for split in ("val", "test"):
        loader = getattr(obj, f"{split}_loader")
        assert sampler_name(loader) == "SequentialSampler", (
            f"{split} 不应打乱，实际 {sampler_name(loader)}"
        )

    # 数据根目录要指向对应的 split 子目录
    for split in ("train", "val", "test"):
        dataset = getattr(obj, f"{split}_dataset")
        assert os.path.basename(os.path.normpath(dataset.data_root)) == split, (
            f"{split}_dataset 的 data_root 指向错误: {dataset.data_root}"
        )

    # batch_size 要透传
    assert obj.train_loader.batch_size == args.batch_size
    print("dataset_init (三份全开): OK")


def case_shared_label_map() -> None:
    """核心不变量：三份数据共用同一个映射，且来自 train.csv。"""
    args = make_args()
    obj = trainer(args)
    obj.dataset_init()

    train_map = obj.train_dataset.get_map()
    expected = build_label_to_index_map(args.train_csv)

    assert train_map == expected, "train 的映射与 build_label_to_index_map(train.csv) 不一致"

    # 必须是同一个对象，而不是内容相等——证明没有重复重建
    assert obj.val_dataset.get_map() is train_map, (
        "val 的映射不是 train 的那个对象，说明被重新构建了"
    )
    assert obj.test_dataset.get_map() is train_map, (
        "test 的映射不是 train 的那个对象，说明被重新构建了"
    )
    print("dataset_init (三份共用 train 映射): OK")


def case_train_only() -> None:
    """只开训练集时，不应创建 val/test 的 dataset 或 loader。"""
    args = make_args(use_val_csv=False, use_test_csv=False)
    obj = trainer(args)

    num_classes = obj.dataset_init()

    assert num_classes == EXPECTED_CLASSES
    assert hasattr(obj, "train_loader"), "缺少 train_loader"
    for split in ("val", "test"):
        assert not hasattr(obj, f"{split}_loader"), (
            f"未启用 {split} 时不应创建 {split}_loader"
        )
    print("dataset_init (仅训练集): OK")


def case_val_without_train_rejected() -> None:
    """未加载训练集却要加载 val，必须报错而不是静默错位。"""
    args = make_args(use_train_csv=False, use_val_csv=True)
    obj = trainer(args)

    try:
        obj.dataset_init()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "未加载训练集时加载 val 应抛 ValueError："
            "否则 val 会自行用 val.csv 建映射，类别下标与训练整体错位"
        )
    print("dataset_init (无训练集时加载 val 报错): OK")


def main() -> int:
    assert os.path.isdir(DATA_ROOT), f"数据集不存在: {DATA_ROOT}"
    case_all_splits()
    case_shared_label_map()
    case_train_only()
    case_val_without_train_rejected()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
