"""一维剖面数据集：CSV -> (2, L) 剖面张量 + 整数标签。

与 BarcodeDataset 的关系
------------------------
两者对外契约完全相同（``__getitem__`` 返回 ``(张量, python int)``），
所以可以互换着塞进同一个训练循环；区别只在张量形状：

    BarcodeDataset  -> (1, H, W) 灰度图
    ProfileDataset  -> (2, L)    剖面（通道 0 平滑、通道 1 高通）

预计算 + 指纹缓存
-----------------
前端 0.077 ms/张，6527 张约 0.5 s，本可以在 __init__ 里每次现算。但仍然
提供 npz 缓存，并要求**缓存带指纹**：一旦剖面前端口径（L / hp_frac /
pre_size）或前端源码改动，旧缓存必须自动失效。

这条规则来自一次真实的坑：另一个项目的数据集只按"文件是否存在"判断
缓存，改了口径仍然命中旧缓存，静默给出旧前端的结果 —— 而这正好会
骗过"用真实链路验证前端改动"这件事本身。

失败策略（显式，不静默）
------------------------
前端返回 None（图太窄）或读图失败时：
    on_fail="error"（默认）: 立刻抛 RuntimeError 并带上路径
    on_fail="skip":          丢弃该样本并计数，训练继续
**不存在"补零当作正常样本"这个选项**：零剖面配真标签等于往训练集里
掺入"全零输入属于 C128"这种错误监督。
"""

import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 前端在 src 顶层（与 datasets 包同级）。这里的导入前提与
# ``from datasets import ...`` 相同：src 已在 sys.path 上。
import profile1d
from profile1d import DEFAULT_L, HP_FRAC, extract_profile

from .image_io import read_barcode_image
from .labels import build_barcode_class_mapping


# 条码类别固定为这两类，保证与 BarcodeDataset 的下标语义一致
# （C128 -> 0, EAN13 -> 1）。不可依赖 CSV 内容排序生成，否则换一份 CSV
# 就可能让两类下标互换。
FIXED_CLASSES = ("C128", "EAN13")

# 缓存的几种结果统计键
CACHE_ARRAY_KEYS = ("P", "labels", "ar", "w", "h")


def frontend_fingerprint(target_length, hp_frac, pre_size):
    """返回剖面前端的指纹，用于判断缓存是否还能用。

    指纹 = sha1(前端源码字节 + 口径参数)。
    把**源码**也算进去，是因为只比对参数无法发现"前端实现被改了但
    参数没变"的情况，而那正是缓存最容易骗人的地方。
    """
    with open(profile1d.__file__, "rb") as handle:
        source = handle.read()
    key = "%d|%.6f|%r" % (int(target_length), float(hp_frac), pre_size)
    digest = hashlib.sha1()
    digest.update(source)
    digest.update(key.encode("utf-8"))
    return digest.hexdigest()[:16]


class ProfileDataset(Dataset):
    """把条码 CSV 变成 (2, L) 剖面样本。

    CSV 必需列: ``image_path``（绝对路径，或配合 data_root 的相对路径）、
    ``label_name``（码制名，内部统一大写后映射成整数）。

    参数:
        csv_file: 样本清单 CSV。
        data_root: 相对路径的基准目录；None 表示 CSV 里就是绝对路径。
        target_length: 剖面长度 L。
        hp_frac: 高通通道的平滑窗口占比。
        pre_size: (H, W) 或 None。给定时**先**把图缩放到该尺寸再投影，
            仅用于"控制组"实验：把横向分辨率压到与 2D 基线一致。
        cache_path: npz 缓存路径；None 表示不落盘。
        on_fail: 见模块 docstring。
        verbose: 是否打印构建过程。
    """

    def __init__(self, csv_file, data_root=None, target_length=DEFAULT_L,
                 hp_frac=HP_FRAC, pre_size=None, cache_path=None,
                 on_fail="error", verbose=True):
        if on_fail not in ("error", "skip"):
            raise ValueError(
                'on_fail 只支持 "error" / "skip"，收到 %r' % (on_fail,)
            )
        self.csv_file = csv_file
        self.data_root = data_root
        self.target_length = int(target_length)
        self.hp_frac = float(hp_frac)
        self.pre_size = pre_size
        self.cache_path = cache_path
        self.on_fail = on_fail
        self.verbose = verbose

        frame = pd.read_csv(csv_file)
        # 与 BarcodeDataset 用同一个映射函数 + 同一份固定类别次序，
        # 保证两边标签下标语义完全一致
        class_names, class_to_idx = build_barcode_class_mapping(
            frame["label_name"].tolist(), FIXED_CLASSES
        )
        self.class_names = list(class_names)
        self.class_to_idx = dict(class_to_idx)
        self.num_classes = len(self.class_names)

        self.paths = [self._resolve(p) for p in frame["image_path"].tolist()]
        self.raw_labels = np.asarray(
            [self.class_to_idx[str(name).strip().upper()]
             for name in frame["label_name"].tolist()],
            dtype=np.int64,
        )

        # 下面四个数组在 _build / _load_cache 里填充
        self.X = None      # (N, 2, L) float32
        self.labels = None  # (N,) int64
        self.ar = None      # (N,) float32 原始 w/h
        self.w = None       # (N,) int32   原始宽度
        self.h = None       # (N,) int32   原始高度
        self.n_fail = 0     # 被丢弃的样本数

        self._build(cache_path)

    # ---------------- 路径与读图 ----------------
    def _resolve(self, path):
        """相对路径用 data_root 拼接，绝对路径原样返回。"""
        path = str(path)
        if self.data_root is not None and not os.path.isabs(path):
            return os.path.join(self.data_root, path)
        return path

    def _read_gray(self, path):
        """读成单通道灰度。复用 image_io 以兼容中文路径。"""
        return read_barcode_image(path, keep_grayscale=True)

    def _try_read(self, path):
        """读图并返回 (image, 失败原因)。不抛异常，由调用方按策略处理。

        为什么要包住读图: read_barcode_image 在文件缺失或无法解码时会抛
        FileNotFoundError。若这里不兜住，"on_fail='skip'"就形同虚设 ——
        一个坏文件会让整个训练在 __init__ 里直接崩掉。
        """
        try:
            return self._read_gray(path), None
        except (FileNotFoundError, OSError, ValueError) as exc:
            return None, str(exc)

    # ---------------- 构建 / 缓存 ----------------
    def _build(self, cache_path):
        """先试缓存，指纹不符或没有缓存就现算。"""
        fingerprint = frontend_fingerprint(
            self.target_length, self.hp_frac, self.pre_size
        )
        self.fingerprint = fingerprint
        if cache_path and os.path.isfile(cache_path):
            if self._load_cache(cache_path, fingerprint):
                return
            if self.verbose:
                print("  剖面缓存指纹不符，重算: %s"
                      % os.path.basename(cache_path))
        self._extract_all()
        if cache_path:
            self._save_cache(cache_path)

    def _load_cache(self, cache_path, fingerprint):
        """读缓存；指纹或形状不符时返回 False，由调用方重算。"""
        with np.load(cache_path, allow_pickle=False) as data:
            stored = str(data["fingerprint"]) if "fingerprint" in data else ""
            if stored != fingerprint:
                return False
            arrays = {key: data[key] for key in CACHE_ARRAY_KEYS}
        n = arrays["P"].shape[0]
        if n != len(self.paths) or arrays["labels"].shape[0] != n:
            return False
        self.X = np.asarray(arrays["P"], np.float32)
        self.labels = np.asarray(arrays["labels"], np.int64)
        self.ar = np.asarray(arrays["ar"], np.float32)
        self.w = np.asarray(arrays["w"], np.int32)
        self.h = np.asarray(arrays["h"], np.int32)
        self.n_fail = len(self.paths) - n
        if self.verbose:
            print("  剖面缓存命中: %s (%d 条)"
                  % (os.path.basename(cache_path), n))
        return True

    def _save_cache(self, cache_path):
        """把结果写盘。用 savez_compressed 省空间（剖面是低频信号，压得动）。"""
        directory = os.path.dirname(os.path.abspath(cache_path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            cache_path,
            P=self.X, labels=self.labels, ar=self.ar, w=self.w, h=self.h,
            fingerprint=np.asarray(self.fingerprint),
        )
        if self.verbose:
            print("  剖面缓存已写: %s (%.1f MB)"
                  % (cache_path, os.path.getsize(cache_path) / 1e6))

    def _extract_all(self):
        """逐张读图 + 提剖面，失败按 on_fail 处理。"""
        total = len(self.paths)
        if self.verbose:
            print("  剖面预计算: %d 张 (L=%d, hp_frac=%.3f, pre_size=%r)"
                  % (total, self.target_length, self.hp_frac, self.pre_size))
        profiles, labels, ars, widths, heights = [], [], [], [], []
        for index, path in enumerate(self.paths):
            image, read_error = self._try_read(path)
            result = None
            if image is not None:
                result = extract_profile(
                    image, L=self.target_length, hp_frac=self.hp_frac,
                    pre_size=self.pre_size,
                )
            if result is None:
                # 读图失败与"图太窄"在这里合并成同一种失败：都无法得到
                # 有效剖面，区别只体现在错误信息里
                if self.on_fail == "error":
                    raise RuntimeError(
                        "剖面提取失败: %s (%s)"
                        % (path, read_error or "图尺寸过小，无法投影")
                    )
                self.n_fail += 1
                if self.verbose and self.n_fail <= 3:
                    print("    [跳过] %s (%s)"
                          % (path, read_error or "尺寸过小"))
                continue
            profile, meta = result
            profiles.append(profile)
            labels.append(int(self.raw_labels[index]))
            ars.append(meta["ar"])
            widths.append(meta["w"])
            heights.append(meta["h"])
            if self.verbose and (index + 1) % 2000 == 0:
                print("    %d/%d" % (index + 1, total))
        if not profiles:
            raise RuntimeError("没有任何样本成功提取剖面: %s" % self.csv_file)
        # 堆成数组；dtype 显式固定，避免下游拿到 float64 白占一倍内存
        self.X = np.asarray(profiles, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.ar = np.asarray(ars, dtype=np.float32)
        self.w = np.asarray(widths, dtype=np.int32)
        self.h = np.asarray(heights, dtype=np.int32)
        if self.verbose:
            print("  剖面预计算完成: 成功 %d / 失败 %d"
                  % (len(self.labels), self.n_fail))

    # ---------------- Dataset 契约 ----------------
    def __len__(self):
        """成功样本数（失败样本不占位）。"""
        return len(self.labels)

    def __getitem__(self, index):
        """返回 (剖面张量 (2, L) float32, 标签 int)。"""
        # asarray 在 dtype 已匹配时零拷贝；直接返回视图会与 self.X 共享
        # 内存，但张量只读，不存在被下游改写的风险
        sample = torch.from_numpy(np.asarray(self.X[index], np.float32))
        return sample, int(self.labels[index])

    # ---------------- 统计 ----------------
    def class_counts(self):
        """各类别样本数，按 class_names 顺序。供逆频率权重使用。"""
        counts = np.bincount(self.labels, minlength=self.num_classes)
        return counts.astype("int64")

    def stats(self):
        """构建统计，写实验记录时直接引用，避免手工数数。"""
        if len(self.labels) == 0:
            return {"n": 0}
        upsample = self.target_length / np.maximum(self.w, 1)
        return {
            "n": int(len(self.labels)),
            "n_fail": int(self.n_fail),
            "class_counts": self.class_counts().tolist(),
            "class_names": list(self.class_names),
            "width_median": int(np.median(self.w)),
            "width_p90": int(np.percentile(self.w, 90)),
            "upsample_median": float(np.median(upsample)),
            "upsample_p10": float(np.percentile(upsample, 10)),
            "target_length": int(self.target_length),
        }
