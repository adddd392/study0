"""条码数据集切分:按源图键分组 + 按码制分层。

为什么要分组切分
----------------
barber.csv 的文件名形如 ``<源图键>_regionNN.png``:同一张照片会被裁成多个
region。若按行随机切，同一张照片的相邻裁剪极可能同时落进 train 和 val，
验证集就"见过"训练集的同一张底图，准确率虚高。所以切分的单位必须是
**源图键**，不是行。

为什么要分层
------------
C128 1504 张、EAN13 5023 张，约 1:3.3。不分层直接切，val/test 里 C128
可能只剩几十张，评估噪声会盖过模型差异。

算法
----
1. 每个类别单独处理(分层)；
2. 把该类的行按源图键聚成若干组；
3. 组内行数作为"不可拆分的最小单位"，用带种子的随机顺序逐个分配到
   当前"离目标还差最多"的 split —— 贪心配平，避免大组扎堆把比例带偏。

同一份 CSV + 同一个 seed 必须得到完全相同的三份文件。

用法::

    python src/split_barcode.py
    python src/split_barcode.py --csv dataset/barber.csv --out-dir dataset
"""

import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

# 三个 split 的名字,顺序与 ratios 一一对应
SPLIT_NAMES = ("train", "val", "test")

# 默认比例
DEFAULT_RATIOS = (0.7, 0.15, 0.15)

# 文件名里表示"同一张源图的第几个裁剪"的后缀
_REGION_SUFFIX = re.compile(r"_region\d+$")


def source_group_key(image_path):
    """从图像路径提取"源图键":同一张底图拍出来的所有裁剪共享这个键。

    规则:取文件名主干,去掉 ``_regionNN`` 后缀。

        照片\\IMAG0393_region00.png -> IMAG0393
        照片\\IMAG0393_region01.png -> IMAG0393   (与上一行同组)
        照片\\EAN13_09_0001_region00.png -> EAN13_09_0001

    注意不要用 ``文件名.split("_")[0]``:那会把 ``20191019_122813_region00``
    和 ``20191019_999999_region00`` 这类完全无关的图归成一组(实测把 6073
    组压成 4961 组，最大组从 10 行涨到 304 行)，虽然不会泄漏，但会让切分
    粒度变粗、比例失真。
    """
    stem = os.path.splitext(os.path.basename(str(image_path)))[0]
    return _REGION_SUFFIX.sub("", stem)


def _group_positions(group_keys):
    """把行号按源图键分组,返回 {源图键: [行号, ...]}。"""
    buckets = {}
    for position, key in enumerate(group_keys):
        buckets.setdefault(key, []).append(position)
    return buckets


def _label_targets(labels, ratios):
    """算出每个 split 在每个类别上应有的行数。

    返回 {split 名: {类别: 目标行数}}，目标 = 比例 × 该类别总行数。
    """
    counts = Counter(labels)
    return {
        name: {label: ratio * total for label, total in counts.items()}
        for name, ratio in zip(SPLIT_NAMES, ratios)
    }


def _assign_groups(buckets, labels, ratios, rng):
    """把每组源图**整体**分配到某个 split，返回 {split 名: [行号, ...]}。

    为什么必须整体分配:一组源图是一个不可拆的最小单位，拆开就等于让同
    一张照片同时出现在两个 split，切分就白做了。

    为什么是全局而非按类别各分一次:实测有 36 个源图键横跨两个类别(同一张
    照片里裁出了 C128 和 EAN13 两种码)。若按类别各切一遍，同一个键会被
    独立地分到两个 split，分组就破了 —— 这一点由 verify_split 兜底发现。

    分配策略:
        组按随机顺序逐个处理，每次选择"缺口最大"的 split。缺口定义为
        该组覆盖到的每个类别上「目标 - 已填」之和，因此能同时兼顾多个
        类别的比例。
    """
    targets = _label_targets(labels, ratios)
    filled = {name: Counter() for name in SPLIT_NAMES}
    assigned = {name: [] for name in SPLIT_NAMES}

    keys = list(buckets.keys())
    order = rng.permutation(len(keys))

    for index in order:
        key = keys[index]
        positions = buckets[key]
        # 这个组在各类别上分别贡献几行
        group_labels = Counter(labels[position] for position in positions)

        best = max(
            SPLIT_NAMES,
            key=lambda name: sum(
                targets[name][label] - filled[name][label]
                for label in group_labels
            ),
        )
        assigned[best].extend(positions)
        filled[best].update(group_labels)
    return assigned


def stratified_group_split(df, ratios=DEFAULT_RATIOS, seed=0,
                           label_col="label_name", path_col="image_path"):
    """按源图键分组、按类别分层切分 DataFrame。

    参数:
        df: 待切分的完整数据表。
        ratios: (train, val, test) 比例，三者之和须为 1。
        seed: 随机种子，固定后结果完全可复现。
        label_col: 类别列名，用于分层。
        path_col: 图像路径列名，用于提取源图键。
    返回:
        {"train": DataFrame, "val": DataFrame, "test": DataFrame}
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios 之和必须为 1，当前 {ratios}")
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"ratios 需要 {len(SPLIT_NAMES)} 个，当前 {len(ratios)}")

    rng = np.random.default_rng(seed)
    labels = df[label_col].astype(str).to_numpy()
    group_keys = [source_group_key(p) for p in df[path_col]]
    buckets = _group_positions(group_keys)

    assigned = _assign_groups(buckets, labels, ratios, rng)
    return {
        # 排序后再取，保证输出行序稳定、diff 可读
        name: df.iloc[sorted(positions)].reset_index(drop=True)
        for name, positions in assigned.items()
    }


def verify_split(splits, full_df, ratios=DEFAULT_RATIOS, path_col="image_path",
                 label_col="label_name", tolerance=0.02):
    """校验切分结果，失败时抛 AssertionError 并说明原因。

    校验项:
        1. 三份行数之和等于原始行数(没有丢行、没有重复)；
        2. 三份的源图键两两不相交(切分的核心目的)；
        3. 每份都包含所有类别(否则分层失败)；
        4. 实际比例与 ratios 的偏差在 tolerance 以内(总体与逐类别各查一次)。

    参数:
        ratios: 本次切分实际请求的比例，校验必须对着它来，不能用默认值。
    """
    total = len(full_df)
    got = sum(len(splits[name]) for name in SPLIT_NAMES)
    assert got == total, f"行数不守恒: {got} != {total}"

    # 没有丢行/重复:三份的 (image_path) 集合之和应等于全集
    all_paths = [
        path
        for name in SPLIT_NAMES
        for path in splits[name][path_col].tolist()
    ]
    assert len(all_paths) == len(set(all_paths)), "存在重复样本"
    assert set(all_paths) == set(full_df[path_col]), "样本集合与原始不一致"

    # 核心不变量:源图键互不相交
    key_sets = {
        name: {source_group_key(p) for p in splits[name][path_col]}
        for name in SPLIT_NAMES
    }
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1:]:
            overlap = key_sets[left] & key_sets[right]
            assert not overlap, (
                f"{left} 与 {right} 共享 {len(overlap)} 个源图键,例如 "
                f"{sorted(overlap)[:3]}"
            )

    # 每份都要有全部类别
    expected_labels = set(full_df[label_col].unique())
    for name in SPLIT_NAMES:
        labels = set(splits[name][label_col].unique())
        assert labels == expected_labels, (
            f"{name} 缺少类别: {expected_labels - labels}"
        )

    # 比例偏差
    got_ratios = {name: len(splits[name]) / total for name in SPLIT_NAMES}
    for name, target in zip(SPLIT_NAMES, ratios):
        assert abs(got_ratios[name] - target) <= tolerance, (
            f"{name} 比例 {got_ratios[name]:.3f} 偏离目标 {target:.3f} 超过 "
            f"{tolerance}"
        )

    # 分层是否奏效:每个类别自己也要按目标比例分布到三份里
    for label in sorted(expected_labels):
        label_total = int((full_df[label_col] == label).sum())
        for name, target in zip(SPLIT_NAMES, ratios):
            got = int((splits[name][label_col] == label).sum())
            actual = got / label_total
            assert abs(actual - target) <= tolerance, (
                f"类别 {label} 在 {name} 中的占比 {actual:.3f} 偏离目标 "
                f"{target:.3f} 超过 {tolerance}"
            )


def describe(splits, label_col="label_name", path_col="image_path"):
    """把三份切分的规模和类别分布整理成可打印的文本。"""
    lines = []
    for name in SPLIT_NAMES:
        part = splits[name]
        groups = len({source_group_key(p) for p in part[path_col]})
        counts = part[label_col].value_counts().sort_index()
        detail = ", ".join(f"{k} {v}" for k, v in counts.items())
        lines.append(
            f"  {name:<5} {len(part):>5} 行 / {groups:>5} 组  ({detail})"
        )
    return "\n".join(lines)


def parse_args():
    """命令行参数。相对路径以当前工作目录为基准。"""
    parser = argparse.ArgumentParser(description="按源图键分组分层切分条码数据集")
    parser.add_argument(
        "--csv", default=os.path.join("dataset", "barber.csv"),
        help="原始 CSV 路径",
    )
    parser.add_argument(
        "--out-dir", default="dataset",
        help="输出目录，三份 CSV 写在这里",
    )
    parser.add_argument(
        "--prefix", default="barcode",
        help="输出文件名前缀，生成 <prefix>_train.csv 等",
    )
    parser.add_argument(
        "--ratios", type=float, nargs=3, default=list(DEFAULT_RATIOS),
        metavar=("TRAIN", "VAL", "TEST"), help="切分比例",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    return parser.parse_args()


def main():
    """读原始 CSV -> 切分 -> 校验 -> 写三份 CSV。"""
    args = parse_args()
    full_df = pd.read_csv(args.csv)
    print(f"原始数据: {args.csv}  {len(full_df)} 行")

    splits = stratified_group_split(
        full_df, ratios=tuple(args.ratios), seed=args.seed
    )
    verify_split(splits, full_df, ratios=tuple(args.ratios))
    print("切分结果(校验通过):")
    print(describe(splits))

    os.makedirs(args.out_dir, exist_ok=True)
    for name in SPLIT_NAMES:
        out_path = os.path.join(args.out_dir, f"{args.prefix}_{name}.csv")
        splits[name].to_csv(out_path, index=False, encoding="utf-8")
        print(f"  已写入 {out_path}")


if __name__ == "__main__":
    main()
