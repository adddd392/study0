"""模型权重的保存与加载。

训练与推理共用的约定集中在这里，避免两处各自硬编码路径与加载参数：

    - 权重以 state_dict 形式保存为 <save_dir>/models/model_epoch_<epoch>.pth
    - **保存前一律把张量搬到 CPU**，落盘的文件不含任何设备相关的 pickle 记录
    - 加载统一 map_location="cpu"，设备迁移交给调用方决定
    - 加载统一 weights_only=True：只接受张量，防止 pickle 反序列化执行任意代码
      （torch 2.4 的 torch.load 默认 weights_only=False，torch 2.6 才改为 True）

"保存前搬到 CPU" 不是洁癖，是必需的：在 DirectML（device 名 dml）上训练的
模型，参数张量属于 privateuseone 设备。直接 torch.save 会把张量序列化成
"torch._utils._rebuild_device_tensor_from_numpy" 这样的重建记录，而加载端
用 weights_only=True 时该记录不在允许列表里，于是抛出 UnpicklingError——
表现就是"训练 10 轮全部跑完，最后加载最佳模型时崩溃"，且 results.txt、
曲线、分类报告都还没写，一次实验的产物全部丢失。
"""

import os

import torch

# 保存目录下存放权重文件的子目录名
MODELS_DIRNAME = "models"


def checkpoint_path(save_dir, epoch):
    """构造某轮次权重的完整路径（不检查文件是否存在）。"""
    return os.path.join(save_dir, MODELS_DIRNAME, f"model_epoch_{epoch}.pth")


def _to_cpu_state_dict(state_dict):
    """把 state_dict 里的张量搬到 CPU 并切断计算图。

    state_dict 的值绝大多数是张量，但官方并不保证（自定义模块可能塞 int、
    str 等），所以非张量原样保留，避免把未知类型改坏。
    """
    return {
        key: (
            value.detach().cpu() if isinstance(value, torch.Tensor) else value
        )
        for key, value in state_dict.items()
    }


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
    # 必须转 CPU 后再存：DML 张量会写出设备相关的重建记录，
    # 加载端 weights_only=True 无法解析（详见模块 docstring）
    torch.save(_to_cpu_state_dict(model.state_dict()), path)
    return path


def load_checkpoint(path):
    """加载 state_dict（映射到 CPU，仅接受张量）。

    返回的权重仍在 CPU 上，需要哪个设备由调用方负责 .to(device)。
    """
    return torch.load(path, map_location="cpu", weights_only=True)
