"""一维剖面模型的独立训练 / 评估脚本。

为什么不改造 train.py
--------------------
``src/train.py`` 是 2D 图像训练器：它假定输入是 ``(N, C, H, W)``、按
``--config`` 取实验配置、并按 2D 的产物格式写 results.txt。本次实验需要
的是**一维输入 + 同一套划分与判据**，两者诉求只有"训练循环"这一小段
重叠，为它去重构核心训练器不划算，且实验正在进行中。

所以这里走"新能力、新脚本"的路线：
    · 复用 metrics（分类报告）、checkpoint（保存/加载）、profiling（MACs）
    · 不复用、也不修改 train.py 的任何代码

产物（写进 --save_path）
------------------------
    results.json                       全部指标 + 配置 + 曲线 + 耗时
    models/model_epoch_<best>.pth      验证损失最小的那一轮
    models/model_last.pth              （--save_last 时）最后一轮
    macs_and_parameters.txt            复杂度口径

用法::

    python src/train_profile.py --train_csv dataset/barcode_test.csv \
        --val_csv dataset/barcode_val.csv \
        --test_csv dataset/barcode_train.csv --num_epochs 10 \
        --save_path .tmp_tests/prof1d_s0 --seed 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 让脚本以 ``python src/xxx.py`` 直接跑时也能 import 同目录模块
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import checkpoint  # noqa: E402
import metrics  # noqa: E402
import models_1d  # noqa: E402
import profiling  # noqa: E402
from datasets.profile import ProfileDataset  # noqa: E402


class SelectChannelsDataset(Dataset):
    """只取剖面的指定通道，用于单通道消融。

    剖面前端默认给两路（0 平滑、1 高通）。这里在 __getitem__ 里做切片，
    而不是重算前端：前端结果与通道选择无关，切一刀就够，也顺手保证
    "消融组的剖面与主口径逐位相同"。
    """

    def __init__(self, base, channels):
        self.base = base
        self.channels = list(channels)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        profile, label = self.base[index]
        return profile[self.channels], label


class AppendArDataset(Dataset):
    """把 ar（原始宽高比）作为附加通道拼进剖面。

    只用于"尺寸捷径"消融：ar 单独就能把多数类基线从 76.9% 推到 91.4%
    （见 Phase 2 诊断），若喂了 ar 精度大涨，说明模型在吃尺寸捷径而不是
    条宽信息。拼成常数通道是最不改动模型结构的做法。
    """

    def __init__(self, base):
        self.base = base
        # ar 归一化到 -1~1：除以全体最大绝对值，避免大 ar 主导前向
        ar = np.asarray(base.ar, dtype=np.float32)
        scale = float(np.max(np.abs(ar))) if ar.size else 1.0
        self.ar = ar / max(scale, 1e-6)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        profile, label = self.base[index]
        # (2, L) -> (3, L)：第三通道是逐位置相同的 ar 常数
        extra = torch.full((1, profile.shape[1]), float(self.ar[index]),
                           dtype=profile.dtype)
        return torch.cat([profile, extra], dim=0), label


def set_seed(seed):
    """固定所有随机源，保证同 seed 可复现（与 2D 侧同一套做法）。"""
    random_seed = int(seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # CPU 侧没有 CUDA 的 deterministic 选项，靠单线程算子 + 固定种子即可


def build_cache_name(split, args):
    """缓存文件名带上全部口径参数，避免不同配置互相覆盖。"""
    tag = "L%d_hp%.3f" % (args.target_length, args.hp_frac)
    if args.pre_size is not None:
        tag += "_pre%dx%d" % (args.pre_size[0], args.pre_size[1])
    return os.path.join(args.cache_dir, "%s_%s.npz" % (split, tag))


def load_split(csv_file, split, args):
    """构建一个划分的剖面数据集（带缓存）。"""
    dataset = ProfileDataset(
        csv_file,
        target_length=args.target_length,
        hp_frac=args.hp_frac,
        pre_size=tuple(args.pre_size) if args.pre_size else None,
        cache_path=build_cache_name(split, args),
        on_fail="error",
        verbose=args.verbose,
    )
    return dataset


def make_loader(dataset, batch_size, shuffle, num_workers):
    """包装 DataLoader。剖面已在内存里，worker 只会复制数据，默认 0。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def evaluate(model, loader, criterion, device, num_classes, class_names):
    """跑一遍评估集，返回 (平均损失, 分类报告)。"""
    model.eval()
    total_loss = 0.0
    total = 0
    targets, predictions = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            # 按样本数加权：最后一个 batch 可能不满，不能简单平均 batch loss
            total_loss += float(loss.item()) * labels.shape[0]
            total += labels.shape[0]
            predictions.extend(logits.argmax(dim=1).tolist())
            targets.extend(labels.tolist())
    report = metrics.classification_report(
        targets, predictions, num_classes, class_names
    )
    return total_loss / max(total, 1), report


def train_one_epoch(model, loader, criterion, optimizer, device):
    """跑一个训练轮，返回平均损失。"""
    model.train()
    total_loss = 0.0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.shape[0]
        total += labels.shape[0]
    return total_loss / max(total, 1)


def parse_args(argv=None):
    """命令行参数。默认值 = 与 2D 基线对齐的那套训练协议。"""
    parser = argparse.ArgumentParser(description="一维剖面模型训练")
    parser.add_argument("--model", default="MobileNetV3-Large-1D",
                        choices=models_1d.available_profile_models())
    parser.add_argument("--train_csv", default="dataset/barcode_train.csv")
    parser.add_argument("--val_csv", default="dataset/barcode_val.csv")
    parser.add_argument("--test_csv", default="dataset/barcode_test.csv")
    parser.add_argument("--target_length", type=int, default=1024)
    parser.add_argument("--hp_frac", type=float, default=0.06)
    parser.add_argument("--pre_size", type=int, nargs=2, default=None,
                        help="控制组：先把图缩到该 (H, W) 再投影")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Adam 的 L2 权重衰减（2D 基线是 0）")
    parser.add_argument("--feed_ar", action="store_true",
                        help="把 ar 作为附加通道拼进剖面（尺寸捷径消融）")
    parser.add_argument("--channels", default="both",
                        choices=("both", "gray", "hp"),
                        help="both=双通道, gray=仅灰度, hp=仅高通（单通道消融）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--cache_dir", default=".tmp_tests/_profile_cache")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    """训练 -> 选最佳轮 -> 测试 -> 写产物。"""
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    print("=" * 78)
    print("  一维剖面实验: %s  seed=%d" % (args.model, args.seed))
    print("=" * 78)
    started = time.time()

    # ---------------- 数据 ----------------
    train_set = load_split(args.train_csv, "train", args)
    val_set = load_split(args.val_csv, "val", args)
    test_set = load_split(args.test_csv, "test", args)
    class_names = list(train_set.class_names)
    num_classes = train_set.num_classes
    print("  样本数: train %d / val %d / test %d"
          % (len(train_set), len(val_set), len(test_set)))
    # 类别分布要在包装成消融数据集之前取：包装类不实现统计接口
    class_counts = train_set.class_counts().tolist()
    print("  类别分布: train %s" % (class_counts,))

    # 通道选择：both(2) / gray(1) / hp(1)
    channel_map = {"both": (0, 1), "gray": (0,), "hp": (1,)}
    channels = channel_map[args.channels]
    if len(channels) != 2:
        train_set = SelectChannelsDataset(train_set, channels)
        val_set = SelectChannelsDataset(val_set, channels)
        test_set = SelectChannelsDataset(test_set, channels)
    in_channels = len(channels)
    print("  输入: %s -> %d 通道" % (args.channels, in_channels))
    if args.feed_ar:
        # ar 消融：多一个常数通道
        train_set = AppendArDataset(train_set)
        val_set = AppendArDataset(val_set)
        test_set = AppendArDataset(test_set)
        in_channels += 1

    train_loader = make_loader(train_set, args.batch_size, True,
                               args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, False,
                              args.num_workers)

    # ---------------- 模型与优化器 ----------------
    model = models_1d.get_profile_model(
        args.model, num_classes=num_classes, in_channels=in_channels
    ).to(args.device)
    n_par = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(train_loader)
    print("  模型参数: %.3f M, 每轮 %d 步" % (n_par / 1e6, steps_per_epoch))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # 2D 基线用的是裸 Adam（weight_decay=0）。这里留一个开关，用于验证
    # "1D 的差距是训练/正则问题而不是信息问题"这个假设
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # ---------------- 训练循环 ----------------
    history = {"train_loss": [], "val_loss": [], "val_top1": [],
               "epoch_seconds": []}
    best = {"epoch": -1, "val_loss": float("inf"), "path": None}
    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        val_loss, val_report = evaluate(
            model, val_loader, criterion, args.device, num_classes, class_names
        )
        elapsed = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(val_report["accuracy"])
        history["epoch_seconds"].append(elapsed)
        print("    Epoch %2d/%d  train %.4f  val %.4f  val Top-1 %.2f%%  %.1fs"
              % (epoch, args.num_epochs, train_loss, val_loss,
                 100.0 * (val_report["accuracy"] or 0.0), elapsed))
        if val_loss < best["val_loss"]:
            previous = best["path"]
            best.update(epoch=epoch, val_loss=val_loss)
            best["path"] = checkpoint.save_checkpoint(
                model, args.save_path, epoch
            )
            # 每改善一次就存一个文件、却不删旧的，一个 run 会堆到 10 个
            # 16 MB 的权重（实测把测试目录撑到 4.3 GB）。这里只保留当前
            # 最佳，旧的立刻删掉。
            if previous and previous != best["path"]:
                try:
                    os.remove(previous)
                except OSError:
                    pass
        if args.save_last:
            checkpoint.save_last_checkpoint(model, args.save_path)

    train_seconds = time.time() - started
    print("  训练完成，最佳轮 %d (val loss %.4f)，用时 %.1f 分钟"
          % (best["epoch"], best["val_loss"], train_seconds / 60.0))

    # ---------------- 用最佳轮做测试 ----------------
    # load_checkpoint 返回的是 state_dict（不是模型），所以要按同一套
    # 架构重建模型再灌权重，避免"存的对象结构"依赖
    model = models_1d.get_profile_model(
        args.model, num_classes=num_classes, in_channels=in_channels
    )
    model.load_state_dict(checkpoint.load_checkpoint(best["path"]))
    model = model.to(args.device)
    test_start = time.time()
    test_loss, test_report = evaluate(
        model, test_loader, criterion, args.device, num_classes, class_names
    )
    test_seconds = time.time() - test_start
    print("  测试: loss %.4f  Top-1 %.2f%%  balanced %.2f%%"
          % (test_loss, 100.0 * (test_report["accuracy"] or 0.0),
             100.0 * (test_report["balanced_accuracy"] or 0.0)))
    print(metrics.format_report(test_report, title="测试集分类报告"))

    # ---------------- 复杂度 ----------------
    input_shape = (1, in_channels, args.target_length)
    macs, params = profiling.get_model_macs_and_parameters(model, input_shape)
    profiling.save_macs_and_parameters(
        macs, params, os.path.join(args.save_path, "macs_and_parameters.txt")
    )

    # ---------------- 写 JSON 产物 ----------------
    results = {
        "config": {
            "model": args.model,
            "seed": int(args.seed),
            "target_length": int(args.target_length),
            "hp_frac": float(args.hp_frac),
            "pre_size": list(args.pre_size) if args.pre_size else None,
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "num_epochs": int(args.num_epochs),
            "label_smoothing": float(args.label_smoothing),
            "weight_decay": float(args.weight_decay),
            "feed_ar": bool(args.feed_ar),
            "channels": args.channels,
            "in_channels": int(in_channels),
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
            "steps_per_epoch": int(steps_per_epoch),
        },
        "n_params": int(n_par),
        "macs": int(macs) if macs is not None else None,
        "class_names": class_names,
        "n_samples": {"train": len(train_set), "val": len(val_set),
                      "test": len(test_set)},
        "class_counts": class_counts,
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "test_loss": float(test_loss),
        "test": test_report,
        "history": history,
        "seconds": {"train": float(train_seconds),
                    "test_eval": float(test_seconds)},
    }
    results_path = os.path.join(args.save_path, "results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print("  已写: %s" % results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
