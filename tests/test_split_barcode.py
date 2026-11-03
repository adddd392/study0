"""split_barcode.py 的验收测试。

契约::

    splits = stratified_group_split(df, ratios=(0.7, 0.15, 0.15), seed=0)

    核心不变量:
    1. 【分组不可拆】同一源图键的所有行必须落在同一个 split 里。
       这是整个切分存在的理由 —— 同一张照片的相邻裁剪若跨了 train/val,
       验证集就"见过"训练集底图,准确率虚高。
    2. 【横跨两类的组也不能拆】实测有 36 个源图键同时含 C128 和 EAN13,
       切分必须按组整体分配,不能按类别各切一遍。
    3. 【可复现】同一个 seed 必须给出完全相同的结果。
    4. 【行数守恒】三份行数之和等于原始行数,无丢行无重复。
    5. 【分层】每个类别都按目标比例分布到三份里。

运行::

    python tests/test_split_barcode.py
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pandas as pd  # noqa: E402

from split_barcode import (  # noqa: E402
    SPLIT_NAMES,
    describe,
    source_group_key,
    stratified_group_split,
    verify_split,
)

# 真实数据(可能不存在,存在时额外跑一次真实规模校验)
REAL_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "barber.csv")
)


def make_fake_df():
    """构造一张程序生成的小表，刻意包含三种情况。

    1. 同一源图的多个 region(IMAG 族两个 region、日期族);
    2. 不同源图不能同组(IMAG0000 vs IMAG0001);
    3. 一个源图横跨两个类别(20140607_200956 同时有 C128 和 EAN13)。

    规模要够大:成组行数相对总量太粗时，比例天然对不齐，会掩盖真正的问题。
    """
    rows = []
    # 30 个 C128 源图，各 2 个 region -> 60 行
    for index in range(30):
        for region in range(2):
            rows.append(
                (rf"照片\IMAG{index:04d}_region{region:02d}.png", "C128")
            )
    # 20 个 EAN13 源图，各 1 个 region -> 20 行
    for index in range(20):
        rows.append((rf"照片\EAN{index:04d}_region00.png", "EAN13"))
    # 一个源图横跨两个类别，必须整组进同一个 split -> 2 行
    rows.append((r"照片\20140607_200956_region00.png", "C128"))
    rows.append((r"照片\20140607_200956_region01.png", "EAN13"))
    return pd.DataFrame(rows, columns=["image_path", "label_name"])


def groups_of(frame):
    """返回某一份数据用到的源图键集合。"""
    return {source_group_key(p) for p in frame["image_path"]}


def case_group_key() -> None:
    """源图键的提取规则。"""
    assert source_group_key(r"a\IMAG0393_region01.png") == "IMAG0393"
    # 日期族必须保留中间那段时间戳，不能被截成 20191019
    assert (
        source_group_key(r"a\20191019_122813_region00.png")
        == "20191019_122813"
    )
    assert (
        source_group_key(r"a\EAN13_09_0001_region00.png") == "EAN13_09_0001"
    )
    # 没有 region 后缀时保持主干不变
    assert source_group_key(r"a\plain.png") == "plain"
    # 不同源图不能同组
    assert source_group_key(r"a\IMAG0001_region00.png") != source_group_key(
        r"a\IMAG0002_region00.png"
    )
    print("源图键提取: OK")


def case_group_not_split() -> None:
    """核心不变量：同一源图键不得跨 split，含横跨两类的组。"""
    df = make_fake_df()
    splits = stratified_group_split(df, ratios=(0.5, 0.25, 0.25), seed=0)
    verify_split(splits, df, ratios=(0.5, 0.25, 0.25), tolerance=0.15)

    # 再显式验一次横跨两类的那个源图
    hit = [
        name for name in SPLIT_NAMES
        if "20140607_200956" in groups_of(splits[name])
    ]
    assert len(hit) == 1, f"横跨两类的源图被拆到了多个 split: {hit}"
    print("源图组不跨 split(含跨类别组): OK")


def case_reproducible() -> None:
    """同一个 seed 必须给出完全相同的结果，换个 seed 应该不同。"""
    df = make_fake_df()
    first = stratified_group_split(df, ratios=(0.5, 0.25, 0.25), seed=0)
    second = stratified_group_split(df, ratios=(0.5, 0.25, 0.25), seed=0)
    for name in SPLIT_NAMES:
        assert first[name]["image_path"].tolist() == (
            second[name]["image_path"].tolist()
        ), f"{name} 在相同 seed 下结果不一致"

    other = stratified_group_split(df, ratios=(0.5, 0.25, 0.25), seed=1)
    changed = any(
        first[name]["image_path"].tolist()
        != other[name]["image_path"].tolist()
        for name in SPLIT_NAMES
    )
    assert changed, "不同 seed 应给出不同切分(否则随机源没生效)"
    print("可复现性: OK")


def case_no_leak_and_balance() -> None:
    """行数守恒、三份互不相交、每份都有全部类别。"""
    df = make_fake_df()
    splits = stratified_group_split(df, ratios=(0.5, 0.25, 0.25), seed=0)

    total = sum(len(splits[name]) for name in SPLIT_NAMES)
    assert total == len(df), f"行数不守恒: {total} != {len(df)}"

    # 三份两两不相交
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1:]:
            assert not (groups_of(splits[left]) & groups_of(splits[right])), (
                f"{left} 与 {right} 共享源图键"
            )

    # 每份都有两类
    for name in SPLIT_NAMES:
        labels = set(splits[name]["label_name"].unique())
        assert labels == {"C128", "EAN13"}, f"{name} 缺类别: {labels}"
    print("互不相交 + 类别齐全: OK")


def case_verify_rejects_leak() -> None:
    """verify_split 必须能发现人为制造的跨 split 泄漏。"""
    df = make_fake_df()
    # 故意把同一个源图的两行拆到两份里
    bad = {
        "train": df.iloc[0:1].reset_index(drop=True),
        "val": df.iloc[1:2].reset_index(drop=True),
        "test": df.iloc[2:].reset_index(drop=True),
    }
    try:
        verify_split(bad, df, tolerance=0.9)
    except AssertionError as exc:
        assert "源图键" in str(exc), f"报错信息没说清原因: {exc}"
    else:
        raise AssertionError("泄漏的切分应被 verify_split 拒绝")
    print("verify_split 拒绝泄漏: OK")


def case_ratios_validated() -> None:
    """比例不合法时要报错，而不是悄悄切错。"""
    df = make_fake_df()
    for bad_ratios in ((0.6, 0.2, 0.2), (0.5, 0.25, 0.15)):
        if abs(sum(bad_ratios) - 1.0) < 1e-9:
            # (0.5,0.25,0.15) 之和为 0.9，走下面的分支
            continue
        try:
            stratified_group_split(df, ratios=bad_ratios, seed=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法比例 {bad_ratios} 应抛 ValueError")
    # 个数不对也要报错
    try:
        stratified_group_split(df, ratios=(0.8, 0.2), seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("比例个数不对时应抛 ValueError")
    print("比例校验: OK")


def case_real_data() -> None:
    """真实 barber.csv 上跑一遍；文件不存在时跳过。"""
    if not os.path.isfile(REAL_CSV):
        print("真实数据切分: 跳过(未找到 dataset/barber.csv)")
        return
    df = pd.read_csv(REAL_CSV)
    splits = stratified_group_split(df, seed=0)
    verify_split(splits, df)

    # 真实数据的关键数字，改动切分逻辑后必须一致
    assert len(splits["train"]) == 4569, (
        f"train 行数变了: {len(splits['train'])}"
    )
    assert len(splits["val"]) == 979
    assert len(splits["test"]) == 979
    print(describe(splits))
    print("真实数据切分: OK")


def main() -> int:
    case_group_key()
    case_group_not_split()
    case_reproducible()
    case_no_leak_and_balance()
    case_verify_rejects_leak()
    case_ratios_validated()
    case_real_data()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
