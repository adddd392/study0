"""可复现性自检：两次独立运行，输出必须完全一致。

用法（任意工作目录均可）::

    python repro_check.py            # 实验组：设种子
    python repro_check.py --no-seed  # 对照组：不设种子

期望：对照组的两次输出不同，实验组的两次输出完全相同。
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets import Mini_ImageNet_Datasets  # noqa: E402
from models import get_model  # noqa: E402
import seeding  # noqa: E402

SEED = 42

# src/ 的上两级才是 代码/，数据放在 代码/data/Mini-ImageNet-Dataset
DATA_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "data", "Mini-ImageNet-Dataset")
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
