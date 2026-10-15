"""Mini-ImageNet 数据集与图像预处理管道。

训练与推理必须共用同一套预处理,否则会出现"训练时一个分布、推理时另一个分布"
的隐性错误。因此这里把归一化常量、预处理管道和标签映射统一抽成模块级函数,
供 train.py / inference.py 复用。
"""

import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# 训练与推理共用的归一化参数。

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def build_eval_transform(inputsize=(224, 224)):
    """构建确定性的评估/推理预处理管道:Resize -> Normalize -> ToTensorV2。

    参数:
        inputsize: (H, W) 目标尺寸。
    返回:
        albumentations.Compose 对象,调用方式为 transform(image=ndarray)['image']。
    """
    return A.Compose(
        [
            # 直接缩放到网络输入尺寸(本实验不含随机裁剪)
            A.Resize(height=inputsize[0], width=inputsize[1]),
            # 归一化到 [-1, 1]
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            # HWC(float32, 0~1) -> CHW(torch.Tensor)
            ToTensorV2(),
        ]
    )


def build_label_to_index_map(csv_file):
    """按训练时的规则构建 label -> index 映射。

    规则:按 label 在 CSV 中**首次出现的顺序**依次编号 0,1,2...

    推理时必须用 train.csv(而不是 val/test.csv)调用本函数,才能得到
    与训练完全一致的下标语义;否则预测出的下标对应的类别会整体错位。

    参数:
        csv_file: 训练集 CSV 路径,需包含 label 列。
    返回:
        dict,形如 {'n01532829': 0, 'n01558993': 1, ...}
    """
    df = pd.read_csv(csv_file)
    label_classes = df["label"].unique()
    return {label: i for i, label in enumerate(label_classes)}


class Mini_ImageNet_Datasets(Dataset):
    def __init__(self, csv_file, data_root, inputsize=(224, 224), label_to_index_map=None):
        self.df = pd.read_csv(csv_file)
        self.data_root = data_root
        self.inputsize = inputsize
        # 复用预先构建好的映射(验证集/测试集使用训练集的映射,保证下标语义一致)
        if label_to_index_map is None:
            self.label_to_index_map = build_label_to_index_map(csv_file)
        else:
            self.label_to_index_map = label_to_index_map
        # 预处理管道只构建一次,避免每个样本重复构造(原实现在 __getitem__ 中逐样本构建)
        self.transform = build_eval_transform(inputsize)

    def get_map(self):
        return self.label_to_index_map

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.data_root, row["label"], row["filename"])
        image = plt.imread(image_path)
        image = self.transform(image=image)["image"]

        label = row["label"]
        label = self.label_to_index_map[label]
        return image, label

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    dataset = Mini_ImageNet_Datasets(
        csv_file=r"..\data\Mini-ImageNet-Dataset\train.csv",
        data_root=r"..\data\Mini-ImageNet-Dataset\train",
    )
    print(len(dataset))
    image, label = dataset[0]
    image = image.permute(1, 2, 0).numpy()
    image = (image * 0.5 + 0.5) * 255
    image = image.astype(np.uint8)
    plt.imshow(image)
    plt.show()
