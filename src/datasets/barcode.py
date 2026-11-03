"""条码码制数据集：CSV -> (图像张量, 整数标签)。

「基础管道恒执行」带来的直接好处：输出形状永远等于 target_size，
与是否提供 transform 无关，忘记传 transform 不会再得到尺寸各异的张量。
"""


import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .image_io import read_barcode_image, read_image_any_path
from .labels import build_barcode_class_mapping
from .transforms import (
    AUGMENT_C39,
    AUGMENT_NONE,
    AUGMENT_SYN,
    build_barcode_preprocess_pipeline,
)


class BarcodeDataset(Dataset):
    """条码码制分类数据集:CSV -> (图像张量, 整数标签)。

    CSV 必需列:
        image_path  : 图像路径。绝对路径直接使用;相对路径配合 data_root。
        label_name  : 类别名(码制),内部统一转成大写再映射成整数。
    CSV 可选列:
        source_domain : 取值 'synthetic' 时该样本走 transform_syn。

    设计要点:
        1. 基础管道(Resize+Normalize+ToTensorV2)恒执行,且只在初始化时
           构建一次。因此输出形状永远等于 target_size,与是否提供 transform
           无关 —— 忘记传 transform 不会再得到尺寸各异的张量。
        2. 增强管道(transform / transform_c39 / transform_syn)只负责改像素,
           不要再包含 ToTensorV2;误包含会立刻抛 TypeError 并指明原因。
        3. 路径、标签、增强类型在初始化时全部预解析成数组,
           __getitem__ 里不再有任何 pandas 开销(旧实现每样本 2 次 iloc)。

    用法:
        dataset = BarcodeDataset(
            csv_file="dataset/barber.csv",
            target_size=(224, 224),
            fixed_classes=("C128", "EAN13"),
        )
        loader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=4,
            worker_init_fn=BarcodeDataset.worker_init_fn,
        )
    """

    # 增强失败的容忍上限:超过就直接报错。
    # 为什么要有上限:旧实现逐张静默降级,transform 写错时训练照跑、
    # loss 照降,但预处理分布已经变了,唯一的线索是几千行警告刷屏。
    MAX_TRANSFORM_FAILURES = 20

    def __init__(self, csv_file, transform=None, target_size=(224, 224),
                 data_root=None, keep_grayscale=True,
                 fixed_classes=("C128", "EAN13"), cache_images=False,
                 transform_c39=None, transform_syn=None,
                 interpolation=cv2.INTER_AREA, verbose=True):
        """初始化数据集。

        参数:
            csv_file: CSV 路径。
            transform: 通用数据增强管道(只改像素,不含 ToTensorV2)。
            target_size: (H, W) 输出尺寸,由基础管道负责实现。
            data_root: 图像根目录;CSV 里是相对路径时用它拼接。
            keep_grayscale: True 输出单通道,False 输出三通道 RGB。
            fixed_classes: 固定类别次序。默认值用 tuple 而不是 list ——
                list 作为默认参数会在所有实例间共享,一旦有人 append
                就会污染其它实例。
            cache_images: 是否把所有原图解码后常驻内存(省每 epoch 的解码)。
            transform_c39: C39 专属增强。仅当 fixed_classes 含 'C39'
                时才可能生效,否则 CSV 里的 C39 样本会先被判为未知类别。
            transform_syn: 合成数据专属增强,优先级高于 transform_c39。
            interpolation: 基础管道缩放插值方式,默认 INTER_AREA。
            verbose: 是否打印数据集概况。
        """
        self.csv_file = csv_file
        self.target_size = tuple(target_size)
        self.data_root = data_root
        self.keep_grayscale = keep_grayscale
        self.verbose = verbose

        # ---- 增强管道:只改像素,基础管道会在最后统一归一化 ----
        self.transform = transform
        self.transform_c39 = transform_c39
        self.transform_syn = transform_syn

        # ---- 读 CSV 并统一标签书写 ----
        self.df = pd.read_csv(csv_file)
        # strip 去掉首尾空白,upper 统一大小写:
        # 否则 'c128 ' 和 'C128' 会被当成两个不同的类
        self.df['label_name'] = (
            self.df['label_name'].astype(str).str.strip().str.upper()
        )

        # ---- 类别映射 ----
        self.class_names, self.class_to_idx = build_barcode_class_mapping(
            label_names=self.df['label_name'].unique(),
            fixed_classes=fixed_classes,
        )
        self.num_classes = len(self.class_names)
        # 把整数标签写回 df:外部可按类别过滤,也可用 df['label_id'] 统计
        self.df['label_id'] = (
            self.df['label_name'].map(self.class_to_idx).astype('int64')
        )

        # ---- 基础预处理管道 ----
        # 只建一次:旧实现每取一个样本都重新 new 一个 A.Compose,
        # 6527 张图乘每个 epoch 都是白白的对象分配开销
        self.base_pipeline = build_barcode_preprocess_pipeline(
            target_size=self.target_size,
            keep_grayscale=keep_grayscale,
            interpolation=interpolation,
        )

        # ---- 预解析:让 __getitem__ 彻底不碰 pandas ----
        self._image_paths = self._resolve_all_paths()
        self._labels = self.df['label_id'].to_numpy()
        self._augment_tags = self._resolve_augment_tags()

        # ---- 可选的图像缓存 ----
        self.cache_images = cache_images
        self._cache = None
        self._transform_failures = 0
        if verbose:
            self._print_summary()
        if cache_images:
            self._preload_images()

    # -- 初始化期的辅助方法 -------------------------------------------------

    def _resolve_all_paths(self):
        """把 CSV 的 image_path 一次性解析成可直接交给 cv2.imread 的列表。

        绝对路径原样返回;相对路径用 data_root 拼接(data_root 为 None 时
        保留原样,由 cv2 相对当前工作目录解析)。
        """
        paths = []
        for raw_path in self.df['image_path']:
            # CSV 读进来可能是 numpy.str_,且可能带首尾空格
            path = str(raw_path).strip()
            if self.data_root is not None and not os.path.isabs(path):
                path = os.path.join(str(self.data_root), path)
            paths.append(path)
        return paths

    def _resolve_augment_tags(self):
        """预先算出每个样本该用哪条增强管道,返回 int8 数组。

        优先级:合成数据(transform_syn) > C39(transform_c39) > 通用。
        未提供对应管道时,该类样本一律回落到 AUGMENT_NONE。
        """
        tags = np.full(len(self._image_paths), AUGMENT_NONE, dtype=np.int8)
        # 合成数据优先级最高,先打标
        if self.transform_syn is not None and (
            'source_domain' in self.df.columns
        ):
            # 统一成小写再比较,避免 'Synthetic' / ' synthetic ' 漏标
            domain = (
                self.df['source_domain'].astype(str).str.strip().str.lower()
            )
            tags[(domain == 'synthetic').to_numpy()] = AUGMENT_SYN
        # C39 只覆盖还没被标成合成数据的样本
        if self.transform_c39 is not None:
            is_c39 = (self.df['label_name'] == 'C39').to_numpy()
            tags[is_c39 & (tags == AUGMENT_NONE)] = AUGMENT_C39
        return tags

    def _print_summary(self):
        """打印数据集概况:类别、映射、各类样本数,便于训练前核对。"""
        counts = self.df['label_name'].value_counts()
        lines = [
            f"\n[BarcodeDataset] {len(self.df)} 张,{self.num_classes} 类",
            f"  CSV: {self.csv_file}",
            f"  class_names: {self.class_names}",
            f"  class_to_idx: {self.class_to_idx}",
            "  各类样本数:",
        ]
        for name in self.class_names:
            lines.append(f"    {name}: {int(counts.get(name, 0))}")
        print("\n".join(lines))

    def _preload_images(self):
        """把所有原图解码一次并常驻内存,省掉每个 epoch 的解码开销。

        注意:缓存的是「未缩放原图」,内存占用取决于原图尺寸。
        本数据集实测约 0.24 MB/张,6527 张约 1.6 GB,开启前请先估算。
        """
        start = time.time()
        if self.verbose:
            print(f"[BarcodeDataset] 预加载 {len(self._image_paths)} 张...")
        # 顺序必须与 _image_paths 一致,__getitem__ 才能直接用下标命中
        self._cache = [self._read_image(p) for p in self._image_paths]
        total_gb = sum(image.nbytes for image in self._cache) / 1e9
        if self.verbose:
            print(f"[BarcodeDataset] 缓存完成: {total_gb:.2f} GB,"
                  f" 耗时 {time.time() - start:.0f}s")

    # -- 单样本读取 ---------------------------------------------------------

    # 读图逻辑放在模块级(见 read_image_any_path / read_barcode_image),
    # 推理端复用同一套,保证训练与推理的读图口径完全一致
    _imread_any_path = staticmethod(read_image_any_path)

    def _read_image(self, img_path):
        """读一张图,并按 keep_grayscale 统一通道数。

        返回:
            keep_grayscale=True  -> (H, W) 单通道 ndarray
            keep_grayscale=False -> (H, W, 3) RGB ndarray
        异常:
            FileNotFoundError: 路径不存在或文件无法解码。
        """
        return read_barcode_image(img_path, self.keep_grayscale)

    def _select_augment(self, idx):
        """返回该样本应使用的增强管道;没有则为 None。"""
        tag = int(self._augment_tags[idx])
        if tag == AUGMENT_SYN:
            return self.transform_syn
        if tag == AUGMENT_C39:
            return self.transform_c39
        return self.transform

    def _apply_augment(self, img, idx):
        """对图像应用该样本选中的增强管道,返回值仍是 ndarray。

        异常处理策略:
            - 前 MAX_TRANSFORM_FAILURES 次:降级为基础预处理,只打一条警告;
            - 超过上限:抛 RuntimeError,让错误在第一批 batch 内暴露,
              而不是让训练带着错误的预处理跑完整个 epoch。
        """
        augment = self._select_augment(idx)
        if augment is None:
            # 没有增强,直接交给基础管道
            return img
        try:
            augmented = augment(image=img)['image']
        except Exception as exc:
            # 捕获面故意放宽:第三方增强库的异常类型不可控,
            # 但用「次数上限 + 首条警告」防止它变成静默降级
            self._handle_augment_failure(exc, idx)
            return img
        # 增强后必须还是 ndarray;拿到张量说明调用方把 ToTensorV2 也塞进来了。
        # 在 try 之外抛,避免被上面的 except 吞掉
        if isinstance(augmented, torch.Tensor):
            raise TypeError(
                "transform / transform_c39 / transform_syn 不应包含 "
                "ToTensorV2:数据集会在增强之后统一执行 "
                "Resize -> Normalize -> ToTensorV2。"
                f" 出错样本: {self._image_paths[idx]}"
            )
        return augmented

    def _handle_augment_failure(self, exc, idx):
        """统计并处理增强失败:超限报错,否则只报一次警。"""
        self._transform_failures += 1
        if self._transform_failures > self.MAX_TRANSFORM_FAILURES:
            raise RuntimeError(
                f"增强管道失败超过 {self.MAX_TRANSFORM_FAILURES} 次,"
                "已停止以避免用错误的预处理继续训练。"
                f"最近一次样本: {self._image_paths[idx]}"
            ) from exc
        if self._transform_failures == 1:
            print(
                f"[BarcodeDataset] 增强失败,已降级为基础预处理:"
                f"{self._image_paths[idx]} ({exc})"
            )
        elif self._transform_failures == self.MAX_TRANSFORM_FAILURES:
            print(
                f"[BarcodeDataset] 增强失败已达 "
                f"{self.MAX_TRANSFORM_FAILURES} 次,再失败将直接报错"
            )

    # -- Dataset 接口 -------------------------------------------------------

    def __len__(self):
        """样本总数。"""
        return len(self._image_paths)

    def __getitem__(self, idx):
        """返回 (image, label)。

        image: float32 张量,(1, H, W) 或 (3, H, W),H/W 等于 target_size。
        label: python int,取值 [0, num_classes)。
        """
        # pandas 支持负下标,但下面的 list 缓存不支持,先统一成正索引
        if idx < 0:
            idx += len(self._image_paths)
        img_path = self._image_paths[idx]

        # 命中缓存就不必再解码,这是 cache_images 唯一的意义
        if self.cache_images:
            img = self._cache[idx]
        else:
            img = self._read_image(img_path)

        # 1) 该样本专属的增强(可能为无)
        img = self._apply_augment(img, idx)

        # 2) 基础管道:恒执行,所以输出形状只由 target_size 决定
        tensor = self.base_pipeline(image=img)['image']

        # 3) 标签从预解析的数组取,避免每样本一次 pandas 索引
        return tensor, int(self._labels[idx])

    # -- 工具方法 -----------------------------------------------------------

    def validate_files(self, limit=None):
        """检查图像是否都在磁盘上,返回缺失/无法读取的路径列表。

        CSV 里存的是绝对路径,数据目录一旦搬家,错误要等 __getitem__
        被调用才暴露。训练前跑一次本方法可以提前发现,避免跑到一半崩。
        """
        paths = self._image_paths if limit is None else (
            self._image_paths[:limit]
        )
        return [path for path in paths if not os.path.isfile(path)]

    def get_map(self):
        """返回 class_to_idx,便于与推理端的映射对齐。"""
        return self.class_to_idx

    @staticmethod
    def worker_init_fn(worker_id):
        """DataLoader 的 worker_init_fn:关掉 cv2 自带的线程池。

        num_workers > 0 时,每个 worker 会自己再开一套 OpenCV 线程,
        和 PyTorch 抢 CPU 核,典型表现是「加了 worker 反而更慢」。
        用法:DataLoader(..., worker_init_fn=BarcodeDataset.worker_init_fn)
        """
        cv2.setNumThreads(0)

    def __repr__(self):
        """调试用的简短描述,不打印逐类统计。"""
        return (
            f"BarcodeDataset(samples={len(self)},"
            f" classes={self.num_classes},"
            f" target_size={self.target_size},"
            f" keep_grayscale={self.keep_grayscale})"
        )
