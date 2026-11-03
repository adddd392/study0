"""数据集预览：跑通「CSV -> 读图 -> 预处理 -> (张量, 标签)」并可视化样本。

它原本是 ``datasets.py`` 底部的 ``__main__`` 块（121 行）。拆包时这部分没有
归属：包里的模块都应该是"可被 import 的库代码"，不该带执行入口。所以独立
成脚本，和 split_barcode.py / inference.py 一样是 src 下的一个入口。

两步检查：

1. Mini-ImageNet：取一个样本，确认整条链路能走通。
2. 条码：逐项断言形状、取值范围、标签范围，再把每个类别各取 2 张排成一行。

第 2 步的图是断言替代不了的：断言只能证明"形状和数值范围对"，证明不了
"标签和图像对得上"。把图和类别名摆在一起，是唯一能一眼看出来的检查方式——
早期版本就出现过标签错位一整个类别、而所有断言都通过的情况。

路径基准取自 configs，不再写死 ``D:\\...`` 或依赖启动目录。

运行::

    python src\\preview_datasets.py             # 弹窗显示样本预览
    python src\\preview_datasets.py --no-show   # 只跑检查，不弹窗
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from configs import BARCODE_DIR, MINI_IMAGENET_ROOT
from datasets import BarcodeDataset, Mini_ImageNet_Datasets


def preview_mini_imagenet(show=True):
    """取 Mini-ImageNet 的第一个样本，确认能读出来。"""
    dataset = Mini_ImageNet_Datasets(
        csv_file=os.path.join(MINI_IMAGENET_ROOT, "train.csv"),
        data_root=os.path.join(MINI_IMAGENET_ROOT, "train"),
    )
    print(f"[Mini-ImageNet] 样本数 = {len(dataset)}")
    image, label = dataset[0]
    print(f"[Mini-ImageNet] 样本0 shape = {tuple(image.shape)}，label = {label}")

    # 归一化到 [-1,1] 的张量：乘 0.5 加 0.5 还原到 [0,1] 才能正常显示
    array = image.permute(1, 2, 0).numpy() * 0.5 + 0.5
    plt.figure()
    plt.title("Mini-ImageNet 样本 0")
    plt.imshow(array)
    plt.axis("off")
    print("[Mini-ImageNet] OK")
    return show


def check_barcode_dataset():
    """构建条码数据集并逐项断言，返回 (dataset, 是否全部通过)。"""
    barcode_csv = os.path.join(BARCODE_DIR, "barber.csv")
    dataset = BarcodeDataset(
        csv_file=barcode_csv,
        target_size=(224, 224),
        # 不需要再传 transform：Resize -> Normalize -> ToTensorV2 这条基础
        # 管道在数据集内部恒执行，输出形状只由 target_size 决定
    )

    # 训练前先确认图像都在盘上：CSV 存的是绝对路径，目录一搬就会抛错
    missing = dataset.validate_files()
    print(f"[条码] 缺失图像 = {len(missing)} 个")
    if missing:
        # 缺图时不做样本检查：直接报清楚缺了多少、第一批例子是什么
        print("[条码] 前 3 个缺失路径:")
        for path in missing[:3]:
            print(f"    {path}")
        print("[条码] 跳过样本检查（先把图像补回磁盘或重跑 split_barcode.py）")
        return dataset, False

    # 样本数必须等于 CSV 的行数，否则说明 CSV 解析或列名有问题
    print(f"[条码] 样本数 = {len(dataset)}")

    # 类别数与映射是标签语义的根：训练和推理必须用同一个映射，
    # 这里打印出来，方便和 train.py 里设置的 num_classes 对照
    print(f"[条码] 类别数 = {dataset.num_classes}")
    print(f"[条码] 类别映射 = {dataset.class_to_idx}")

    # 打印各类样本数：若某一类为 0，说明 CSV 里该类样本被漏读
    counts = dataset.df["label_name"].value_counts().to_dict()
    print(f"[条码] 类别分布 = {counts}")

    # 取第 0 个样本，跑通「读图 + 预处理 + 取标签」整条链路
    sample_image, sample_label = dataset[0]
    print(f"[条码] 样本0 shape = {tuple(sample_image.shape)}")
    print(f"[条码] 样本0 dtype = {sample_image.dtype}")
    print(f"[条码] 样本0 label = {sample_label}")

    # keep_grayscale 默认 True，所以是单通道：(C, H, W) = (1, 224, 224)。
    # 若之后要喂 3 通道主干（如 ResNet），需要传 keep_grayscale=False，
    # 期望值随之变成 (3, 224, 224)
    assert tuple(sample_image.shape) == (1, 224, 224), (
        f"期望 (1, 224, 224)，实际 {tuple(sample_image.shape)}"
    )

    # 预处理后必须是浮点张量（Normalize + ToTensorV2 的结果）
    assert sample_image.dtype.is_floating_point, (
        f"期望浮点张量，实际 {sample_image.dtype}"
    )

    # Normalize 用 mean=std=0.5 把 0~255 映射到 -1~1，越界说明常量配错。
    # eps 只用来吸收浮点误差，不放宽语义
    eps = 1e-5
    assert float(sample_image.min()) >= -1.0 - eps, "归一化下界越界"
    assert float(sample_image.max()) <= 1.0 + eps, "归一化上界越界"

    # 标签必须落在 [0, num_classes)，且能反查出类别名：
    # 这条不成立的话，训练能跑但每个 batch 的监督信号都是错的
    assert 0 <= sample_label < dataset.num_classes, f"标签越界:{sample_label}"
    print(f"[条码] 标签反查 = {dataset.class_names[sample_label]}")
    print("[条码] 通过：barber.csv 导入正常")
    return dataset, True


def show_barcode_samples(dataset):
    """每个类别各取 2 张，排成一行显示。"""
    # 中文字体：matplotlib 默认字体没有汉字，不加这行标题会显示成方块
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    # 负号用 ASCII 显示，避免部分字体把负号渲染成缺字方块
    plt.rcParams["axes.unicode_minus"] = False

    # 逐类别挑前 2 个下标：按类别取样，才能同时看到两种码制
    show_indices = []
    for class_name in dataset.class_names:
        in_class = dataset.df["label_name"] == class_name
        rows = dataset.df.index[in_class].tolist()
        show_indices.extend(rows[:2])

    # 列数随实际取到的张数变化，避免某一类样本不足时子图空着
    count = len(show_indices)
    if count == 0:
        return
    fig, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.4))
    # count == 1 时 axes 是单个对象而不是数组，统一成可迭代
    axes = np.atleast_1d(axes)
    for ax, index in zip(axes, show_indices):
        image, label = dataset[index]
        # (1, H, W) 且归一化到 [-1,1] -> (H, W) 且 [0,1] 才能直接显示：
        # squeeze 去掉通道维，x * 0.5 + 0.5 是 (x / 255 - 0.5) / 0.5 的逆变换
        array = image.squeeze(0).numpy() * 0.5 + 0.5
        # 不写死 vmin/vmax：让 matplotlib 按每张图的实际范围拉伸对比度。
        # 条码照片普遍偏暗（归一化后常落在 0.0x~0.4），写死 0~1 会糊成一片；
        # 真实数值范围由上面对 min/max 的断言负责，不靠这张图判断
        ax.imshow(array, cmap="gray")
        # 标题写成「行号 + 类别名」：行号可用于回到 CSV 逐行核对
        ax.set_title(f"#{index}  {dataset.class_names[label]}", fontsize=10)
        # 关掉坐标轴：条码图的像素坐标没有信息量，留着只占地方
        ax.axis("off")
    fig.suptitle(f"BarcodeDataset 样本预览（共 {count} 张）", fontsize=12)
    fig.tight_layout()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="数据集预览：检查链路并可视化样本"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只跑检查、不弹窗（无图形界面环境用）",
    )
    args = parser.parse_args(argv)

    preview_mini_imagenet()
    dataset, passed = check_barcode_dataset()
    if passed:
        show_barcode_samples(dataset)

    # plt.show() 放到最后：先让检查结果打印完，否则弹窗未关闭前看不到输出
    if not args.no_show:
        plt.show()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
