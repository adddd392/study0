"""profile1d.py 的黄金测试。

锁定六件事：
    1. 列投影保持宽度、且等于"每列沿行的平均值"
    2. 线性重采样严格保端点、保线性
    3. 高通能消掉基线漂移（加一条巨大线性斜坡不改变结果）
    4. 输出的形状 / 类型 / 取值范围符合契约
    5. 过窄图像返回 None；全黑全白图不产生 NaN
    6. pre_size 控制组只改"横向分辨率"，不改 ar

运行::

    python tests/test_profile1d.py
"""

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from profile1d import (  # noqa: E402
    HP_FRAC,
    column_profile,
    extract_profile,
    high_pass,
    hp_clip_fraction,
    normalize_gray,
    resize_1d,
    to_gray,
)


def _square_wave(width=60, bar=3, high=255, length=40):
    """构造一张竖直条码图: 每 bar 列黑、bar 列白, 交替。"""
    img = np.zeros((length, width), np.uint8)
    for x in range(width):
        # (x // bar) 为偶数段填亮, 奇数段保持黑 -> 竖直条
        img[:, x] = high if (x // bar) % 2 == 0 else 0
    return img


def case_to_gray() -> None:
    """三通道 BGR 图应被转成单通道, 二维图应原样返回。"""
    bgr = np.zeros((10, 12, 3), np.uint8)
    bgr[:, :, 2] = 255          # 纯红, 转灰度后亮度约 76
    gray = to_gray(bgr)
    assert gray.ndim == 2 and gray.shape == (10, 12), "三通道转灰度形状不对"

    already = np.zeros((10, 12), np.uint8)
    assert to_gray(already).shape == (10, 12), "二维图不应被改动"

    try:
        to_gray(np.zeros((2, 3, 4, 5), np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError("非法维度应抛 ValueError")

    try:
        to_gray(None)
    except ValueError:
        pass
    else:
        raise AssertionError("None 输入应抛 ValueError")
    print("to_gray: OK")


def case_column_profile_mean() -> None:
    """列投影必须等于"该列所有行的平均", 且长度等于宽度。"""
    img = np.zeros((8, 20), np.uint8)
    # 每列一个固定值, 行与行之间完全相同 -> 投影应精确等于该值
    img[:, :] = np.arange(20, dtype=np.uint8)[None, :]
    prof = column_profile(img)
    assert prof.shape == (20,), "投影长度应等于宽度, 实际 %r" % (prof.shape,)
    expected = np.arange(20, dtype=np.float32)
    assert np.allclose(prof, expected), "投影不是列均值"
    print("column_profile 列均值: OK")


def case_column_profile_vertical_bars() -> None:
    """竖直条的投影应还原出"黑-白-黑"的方波。"""
    img = _square_wave(width=60, bar=3)
    prof = column_profile(img)
    # 前 3 列是亮段, 接下来 3 列是暗段
    assert np.allclose(prof[0:3], 255.0), "亮段投影不为 255"
    assert np.allclose(prof[3:6], 0.0), "暗段投影不为 0"
    print("column_profile 竖直条方波: OK")


def case_resize_1d_linear() -> None:
    """线性重采样: 保端点、保线性、长度正确。"""
    src = np.arange(100, dtype=np.float32)
    out = resize_1d(src, 200)
    assert out.shape == (200,), "长度应为 200"
    assert abs(float(out[0]) - 0.0) < 1e-4, "起点未对齐"
    assert abs(float(out[-1]) - 99.0) < 1e-4, "终点未对齐"
    # 中点应落在 49.7487 (100/199 * 99)
    assert abs(float(out[100]) - 100.0 / 199.0 * 99.0) < 1e-3, "中点插值不对"
    # 同长度时应返回副本, 不共享内存
    same = resize_1d(src, 100)
    assert same.shape == (100,)
    same[0] = -1.0
    assert src[0] == 0.0, "同长度分支返回了共享内存的视图"
    print("resize_1d 线性: OK")


def case_high_pass_removes_baseline() -> None:
    """加上巨大线性基线后, 高通输出的内部应完全不变。"""
    n = 400
    x = np.arange(n, dtype=np.float64)
    square = np.where((x // 7) % 2 == 0, 200.0, 40.0)
    hp_plain = high_pass(square)
    # 基线幅度远大于信号幅度: 不改高通就说明基线被吃掉了
    hp_ramp = high_pass(square + x * 2.0)
    k = max(3, int(round(HP_FRAC * n)))
    half = k // 2 + 2                    # 两端 reflect 填充区不计入比较
    diff = np.abs(hp_plain[half:-half] - hp_ramp[half:-half]).max()
    assert diff < 1e-6, "基线未被完全消除, 最大残差 %g" % diff
    # 常数剖面经高通后应恒为 0
    assert np.allclose(high_pass(np.full(64, 128.0)), 0.0, atol=1e-9)
    print("high_pass 去基线: OK")


def case_normalize_range() -> None:
    """灰度剖面归一化后必须落在 [-1, 1]。"""
    raw = np.array([0.0, 127.5, 255.0], np.float32)
    out = normalize_gray(raw)
    assert np.allclose(out, [-1.0, 0.0, 1.0]), "归一化结果不是 [-1,0,1]"
    print("normalize_gray: OK")


def case_extract_contract() -> None:
    """extract_profile 的形状 / 类型 / meta 契约。"""
    img = _square_wave(width=300, bar=6, length=120)
    P, meta = extract_profile(img, L=512)
    assert P.shape == (2, 512), "剖面形状应为 (2, 512), 实际 %r" % (P.shape,)
    assert P.dtype == np.float32, "dtype 应为 float32"
    assert np.all(np.isfinite(P)), "剖面含 NaN/Inf"
    # 通道 0 是固定口径归一化, 必须落在 [-1, 1]
    assert float(P[0].min()) >= -1.0 - 1e-6, "通道0 下越界"
    assert float(P[0].max()) <= 1.0 + 1e-6, "通道0 上越界"
    # 通道 1 截断到 [-1, 1]
    assert float(np.abs(P[1]).max()) <= 1.0 + 1e-6, "通道1 未截断"
    assert abs(meta["ar"] - 300.0 / 120.0) < 1e-9, "ar 计算错误"
    assert meta["w"] == 300 and meta["h"] == 120, "meta 尺寸错误"
    assert meta["L"] == 512, "meta L 错误"
    print("extract_profile 契约: OK")


def case_extract_rejects_narrow() -> None:
    """宽度低于门槛时应返回 None, 而不是抛异常或返回垃圾。"""
    assert extract_profile(np.zeros((40, 5), np.uint8)) is None, "过窄未拒绝"
    assert extract_profile(np.zeros((2, 400), np.uint8)) is None, "过矮未拒绝"
    ok = extract_profile(np.zeros((40, 200), np.uint8))
    assert ok is not None, "正常尺寸被误判为失败"
    print("extract_profile 拒绝过小图: OK")


def case_extract_flat_images() -> None:
    """全黑 / 全白图不能产生 NaN（除零陷阱）。"""
    for value in (0, 255):
        img = np.full((60, 200), value, np.uint8)
        P, _ = extract_profile(img, L=256)
        assert np.all(np.isfinite(P)), "全 %d 图产生了 NaN/Inf" % value
        # 常数剖面 -> 高通应恒为 0
        assert np.allclose(P[1], 0.0, atol=1e-6), "常数图的高通不为 0"
    print("extract_profile 全黑全白: OK")


def case_extract_deterministic() -> None:
    """同一张图两次提取必须逐位相同（前端不能有随机性）。"""
    img = _square_wave(width=500, bar=4, length=80)
    a, _ = extract_profile(img, L=1024)
    b, _ = extract_profile(img, L=1024)
    assert np.array_equal(a, b), "前端不是确定性的"
    print("extract_profile 确定性: OK")


def case_pre_size_control_group() -> None:
    """pre_size 应压掉横向细节, 但不改 ar（控制组的核心契约）。"""
    # 600 宽的细条纹(周期 4), 缩到 20 宽后会被面积平均抹平
    img = np.zeros((100, 600), np.uint8)
    for x in range(600):
        img[:, x] = 255 if (x // 2) % 2 == 0 else 0
    fine, meta_fine = extract_profile(img, L=256)
    coarse, meta_coarse = extract_profile(img, L=256, pre_size=(50, 20))
    # 原始图投影是清晰方波, 缩放后应接近常数
    assert float(fine[0].std()) > 0.5, "原始投影方差过小, 构造有问题"
    assert float(coarse[0].std()) < 0.1, "pre_size 没有压掉横向细节"
    # ar 必须取缩放前的原图尺寸
    assert abs(meta_fine["ar"] - meta_coarse["ar"]) < 1e-9, "pre_size 影响了 ar"
    assert meta_coarse["w"] == 600 and meta_coarse["h"] == 100, "meta 用了缩放后尺寸"
    print("pre_size 控制组: OK")


def case_clip_fraction_range() -> None:
    """截断比例统计必须落在 [0, 1] 且能反映极值。"""
    img = _square_wave(width=300, bar=6, length=120)
    P, _ = extract_profile(img, L=512)
    frac = hp_clip_fraction(P)
    assert 0.0 <= frac <= 1.0, "截断比例越界: %r" % frac
    # 人为构造全为极值的通道, 比例应为 1
    fake = np.zeros((2, 10), np.float32)
    fake[1] = 1.0
    assert abs(hp_clip_fraction(fake) - 1.0) < 1e-9, "全截断样本的比例应为 1"
    print("hp_clip_fraction: OK")


def main() -> int:
    case_to_gray()
    case_column_profile_mean()
    case_column_profile_vertical_bars()
    case_resize_1d_linear()
    case_high_pass_removes_baseline()
    case_normalize_range()
    case_extract_contract()
    case_extract_rejects_narrow()
    case_extract_flat_images()
    case_extract_deterministic()
    case_pre_size_control_group()
    case_clip_fraction_range()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
