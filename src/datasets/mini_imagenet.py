"""Mini-ImageNet 数据集（CSV 索引 + 按类别分目录的图像）。

读者注意：这里用 matplotlib.pyplot.imread 读图而不是 read_image_any_path，
历史原因保留。两者的差别在中文路径兼容性上，见 image_io 模块的说明。
"""


import os

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset

from .labels import build_label_to_index_map
from .transforms import build_eval_transform


class Mini_ImageNet_Datasets(Dataset):
    def __init__(self, csv_file, data_root, inputsize=(224, 224),
                 label_to_index_map=None):
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
        image_path = os.path.join(
            self.data_root, row["label"], row["filename"])
        image = plt.imread(image_path)
        image = self.transform(image=image)["image"]

        label = row["label"]
        label = self.label_to_index_map[label]
        return image, label

    def __len__(self):
        return len(self.df)
