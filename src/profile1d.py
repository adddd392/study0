"""一维条码剖面前端（文件名刻意不用 profile，避免遮蔽标准库）：从灰度图得到定长一维剖面。

为什么需要一维剖面
------------------
2D CNN 的输入是 (224, 224) 的灰度图。本模块把同一张图变成 (2, L) 的
一维信号，用来回答一个此前没有数据的问题：**2D 布局到底贡献了多少精度**。

口径（由可行性探针实测确定，写死，不要随意改）
--------------------------------------------
1. 我们的条码图有 98% 是**竖直条**（梯度主峰落在 0 度），所以沿"行"
   平均（axis=0）得到的列投影就是最锐利的剖面；若沿列平均，投影锐度
   只有 1/10.6，条与空会糊成一片。
2. 只沿 **x 方向** 重采样：横向是条宽信息所在，纵向被平均掉。
3. 归一化沿用 2D 基线的固定口径 `(x / 255 - 0.5) / 0.5 -> [-1, 1]`，
   不做逐样本标准化 —— 逐样本标准化会抹掉"对比度"这个真实信息。
4. 高通通道的平滑窗口占比与增益都是固定常量（见 HP_FRAC / HP_GAIN），
   固定而非按样本自适应，保证同一张图永远得到同一份输入。

纯函数、不依赖 torch
--------------------
前端可被 numpy / sklearn 基线直接复用，避免"CNN 一套前端、基线另一套
前端"这种最常见的口径漂移。
"""

import cv2
import numpy as np


# 剖面长度的默认值。原图宽度中位数 331、p90 1003、最大 2305：
# L=1024 意味约 90% 的样本是"上采样"，不会丢横向细节。
DEFAULT_L = 1024

# 投影的最小尺寸门槛。低于这个宽/高时条与空的边界已经糊在一起，
# 再投影也提不出有效信号，直接判为失败，由数据集决定是丢弃还是报错。
MIN_WIDTH = 12
MIN_HEIGHT = 6

# 高通通道的平滑窗口占比：窗口长度 = round(HP_FRAC * L)。
# 0.06 在 L=1024 时约 61 像素，正好抹掉"条"级别的起伏，只留下
# "整片偏黑 / 整片偏白"这种基线漂移，减掉之后剩下的就是条边缘。
HP_FRAC = 0.06

# 高通通道的固定增益。相减后幅度天然很小（相邻条宽的灰度差只有
# 几级），乘一个增益再截断到 [-1, 1]，与灰度通道同量纲。
#
# 这个值是**标定出来的**，不是猜的：在 200 张训练图上统计 |高通| 的
# 分位数（p50 0.2015 / p90 0.4820 / p99 0.7238 / p99.9 0.9091），
# 增益 1.4 让 p99 刚好落到 1.0，实测截断比例 1.10% —— 既没削平边缘，
# 通道幅度也不至于太小。作为对照：增益 2.0 会截断 8.6%，4.0 会截断 41%
# （初版就是 4.0，标定后改掉）。
HP_GAIN = 1.4


def to_gray(img):
    """把 (H, W) 或 (H, W, 3) 图像统一成单通道。

    cv2 读到的是 BGR 三通道，条码判别只用亮度，转灰度可以少一倍计算。
    """
    if img is None:
        raise ValueError("img 为 None")
    if img.ndim == 2:
        # 已经是单通道，原样返回（不做 copy，调用方不应原地修改）
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError("不支持的图像形状: %r" % (img.shape,))


def column_profile(gray):
    """沿行平均，得到长度 = 宽度的列投影。

    竖直条码在每一行上的黑白排列几乎相同，沿行平均等于把相同的信号
    叠加 (h) 次 —— 这是免费的降噪：随机噪声按 sqrt(h) 衰减，而条的
    边缘是各行共有的，会被完整保留。
    """
    if gray.ndim != 2:
        gray = to_gray(gray)
    # 用 float64 累加，避免 uint8 求和溢出，再降回 float32 省内存
    prof = gray.astype(np.float64).mean(axis=0)
    return prof.astype(np.float32)


def _box_smooth(prof, k):
    """长度 k 的滑动平均，边界用 reflect 填充。

    为什么边界必须是 reflect:
        条码两端是静区（纯白）与护线。若用 zero 填充，减均值之后会
        在两端凭空造出一个巨大的假边缘，正好盖掉护线信息 ——
        而护线恰恰是区分 EAN13 与 Code128 的关键线索之一。
    """
    k = int(k)
    if k < 3:
        # 窗口太小没有平滑意义，直接返回副本（保持"不原地改"的契约）
        return prof.astype(np.float64, copy=True)
    if k % 2 == 0:
        # 只有奇数窗口才有对称的中心点，否则平滑结果会整体偏移半格
        k += 1
    half = k // 2
    padded = np.pad(prof.astype(np.float64), half, mode="reflect")
    kernel = np.ones(k, dtype=np.float64) / float(k)
    # mode="valid" 保证输出长度与输入一致（两端各吃掉 half）
    return np.convolve(padded, kernel, mode="valid")


def high_pass(prof, frac=HP_FRAC):
    """去掉缓慢变化的基线，只保留条/空的边缘。

    条码图常见两种基线漂移：光照不均（一侧亮一侧暗）、纸张老化发黄。
    它们只改变"整体亮度水平"，不改变条宽比例，属于干扰项，减掉即可。
    """
    n = prof.shape[0]
    # 窗口至少 3，否则等于没平滑
    k = max(3, int(round(frac * n)))
    smooth = _box_smooth(prof, k)
    return prof.astype(np.float64) - smooth


def resize_1d(prof, L):
    """线性重采样到长度 L。

    为什么用线性插值:
        条码的判别信息是"条宽与空宽的比例"。线性插值对近似方波的
        宽度扰动最小；NEAREST 会整条丢掉细边（与 2D 侧在缩小场景
        选 INTER_AREA 是同一个理由）。
    """
    n = prof.shape[0]
    if n == L:
        return prof.astype(np.float32, copy=True)
    # 归一化坐标上做插值：源/目标长度不同也能严格对齐首尾
    src_x = np.linspace(0.0, 1.0, n)
    dst_x = np.linspace(0.0, 1.0, L)
    return np.interp(dst_x, src_x, prof).astype(np.float32)


def normalize_gray(prof):
    """灰度剖面 0~255 -> [-1, 1]，与 2D 基线的 Normalize(0.5, 0.5) 一致。"""
    return (prof.astype(np.float32) / 255.0 - 0.5) / 0.5


def extract_profile(img, L=DEFAULT_L, hp_frac=HP_FRAC, pre_size=None):
    """原图 -> (P[2, L] float32, meta)；尺寸过小时返回 None。

    参数:
        img: 原始灰度图或 BGR 图。
        L: 剖面长度。
        hp_frac: 高通平滑窗口占比。
        pre_size: (H, W) 或 None。给定时**先**把图缩放到该尺寸再投影。
            这个参数只用于"控制组"实验：把横向分辨率压到与 2D 基线
            一致，从而把"分辨率差异"与"有没有纵向维度"两个因素拆开。

    返回:
        (P, meta)。P 的通道 0 是平滑剖面，通道 1 是高通剖面；
        meta 里 ar 取**缩放前**原图的 w/h（3D 实验里它是尺寸特征）。
    """
    gray = to_gray(img)
    # 原始尺寸与高宽比：必须在使用 pre_size 之前取，否则控制组里
    # ar 会变成缩放后尺寸的函数，与主实验不可比
    h0, w0 = gray.shape
    ar = float(w0) / float(h0)
    if w0 < MIN_WIDTH or h0 < MIN_HEIGHT:
        # 太窄/太矮，投影没有意义，交给调用方按策略处理
        return None
    if pre_size is not None:
        # cv2.resize 的参数顺序是 (宽, 高)，与 numpy 的 (h, w) 相反
        gray = cv2.resize(gray, (int(pre_size[1]), int(pre_size[0])),
                          interpolation=cv2.INTER_AREA)
    # 1) 沿行平均 -> 列投影（保留横向条宽信息，纵向被平均掉）
    prof = column_profile(gray)
    # 2) 重采样到固定长度，使不同宽度的样本可以拼进同一个 batch
    prof = resize_1d(prof, L)
    # 3) 通道 0：固定口径归一化，与 2D 基线对齐
    ch_gray = normalize_gray(prof)
    # 4) 通道 1：去基线后乘固定增益并截断，避免极端值主导 loss
    ch_hp = high_pass(ch_gray, hp_frac) * HP_GAIN
    ch_hp = np.clip(ch_hp, -1.0, 1.0).astype(np.float32)
    # stack 后得到 (2, L)；dtype 统一 float32，与 2D 的 ToTensorV2 一致
    P = np.stack([ch_gray.astype(np.float32), ch_hp], axis=0)
    meta = {
        "ar": ar,          # 原始 w/h，供"尺寸捷径"消融使用
        "w": w0,           # 原始宽度，用于统计"上采样/降采样"比例
        "h": h0,           # 原始高度
        "L": int(L),       # 实际使用的剖面长度
        "hp_frac": float(hp_frac),
    }
    return P, meta


def hp_clip_fraction(P):
    """返回高通通道被截断到边界的比例，用于标定 HP_GAIN。

    比例过高说明增益太大、信息被削平；接近 0 说明增益太小、
    通道幅度不足。只做统计，不参与前向计算。
    """
    hp = P[1].astype(np.float64)
    if hp.size == 0:
        return 0.0
    hit = np.count_nonzero(np.abs(hp) >= 1.0 - 1e-6)
    return float(hit) / float(hp.size)
