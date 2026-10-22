"""模型权重的保存与加载。

训练与推理共用的约定集中在这里，避免两处各自硬编码路径与加载参数：

    - 权重以 state_dict 形式保存为 <save_dir>/models/model_epoch_<epoch>.pth
    - 加载统一 map_location="cpu"，设备迁移交给调用方决定
    - 加载统一 weights_only=True：只接受张量，防止 pickle 反序列化执行任意代码
      （torch 2.4 的 torch.load 默认 weights_only=False，torch 2.6 才改为 True）
"""

import os

import torch

# 保存目录下存放权重文件的子目录名
MODELS_DIRNAME = "models"


def checkpoint_path(save_dir, epoch):
    """构造某轮次权重的完整路径（不检查文件是否存在）。"""
    return os.path.join(save_dir, MODELS_DIRNAME, f"model_epoch_{epoch}.pth")


def save_checkpoint(model, save_dir, epoch):
    """保存模型权重，返回写入的文件路径。

    Args:
        model: 待保存的模型
        save_dir: 本次训练的保存目录（其下会创建 models/ 子目录）
        epoch: 轮次编号，用于文件名

    Returns:
        写入的 .pth 文件完整路径
    """
    path = checkpoint_path(save_dir, epoch)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(path):
    """加载 state_dict（映射到 CPU，仅接受张量）。

    返回的权重仍在 CPU 上，需要哪个设备由调用方负责 .to(device)。
    """
    return torch.load(path, map_location="cpu", weights_only=True)
