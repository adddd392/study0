"""可复现性自检：两次独立运行，输出必须完全一致。

这不是单元测试，而是「对照实验」工具：

    对照组（不设种子）跑两次 → 输出必须【不同】
    实验组（设种子）  跑两次 → 输出必须【相同】

只有对照组的两次不同、实验组的两次相同，才能证明种子真的生效。

运行::

    python tests/test_reproducibility.py --no-seed > a1.txt
    python tests/test_reproducibility.py --no-seed > a2.txt
    python tests/test_reproducibility.py           > b1.txt
    python tests/test_reproducibility.py           > b2.txt

    fc /b a1.txt a2.txt     # 期望：有差异
    fc /b b1.txt b2.txt     # 期望：no differences
"""

import argparse
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import seeding  # noqa: E402
from datasets import Mini_ImageNet_Datasets  # noqa: E402
from models import get_model  # noqa: E402

SEED = 42

# src/ 的上两级才是 代码/，数据放在 代码/data/Mini-ImageNet-Dataset
DATA_ROOT = os.path.abspath(
    os.path.join(_SRC, "..", "..", "data", "Mini-ImageNet-Dataset")
)


def main() -> int:
    parser = argparse.ArgumentParser(description="可复现性自检")
    parser.add_argument(
        "--no-seed", action="store_true", help="对照组：不设随机种子"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader 进程数"
    )
    args = parser.parse_args()

    if not args.no_seed:
        seeding.set_seed(SEED)

    print(f"seed_used = {not args.no_seed}")
    print(f"num_workers = {args.num_workers}")

    # ---- 1) 模型权重初始化是否可复现 ----
    model = get_model("Resnet-50", 10)
    weight = model.fc.weight.detach()
    print(f"fc.weight.sum   = {weight.sum().item():.10f}")
    print(f"fc.weight[0,:5] = {weight[0, :5].tolist()}")

    # ---- 2) DataLoader 取样顺序是否可复现 ----
    dataset = Mini_ImageNet_Datasets(
        csv_file=os.path.join(DATA_ROOT, "train.csv"),
        data_root=os.path.join(DATA_ROOT, "train"),
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=args.num_workers,
    )
    labels: list[int] = []
    for _, targets in loader:
        labels.extend(targets.tolist())
        if len(labels) >= 12:
            break
    print(f"前 12 个标签    = {labels[:12]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
