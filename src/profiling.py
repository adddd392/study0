"""
模型 MACs 和参数量统计工具
"""

import os

import torch
from thop import profile


def get_model_macs_and_parameters(model, input_shape):
    """统计模型乘加次数（macs）与参数量，并写入报告文件。

    注意：thop.profile 返回的是 macs（乘加次数），不是 FLOPs。
    按惯例 1 mac ≈ 2 FLOPs，需要 FLOPs 时自行乘 2。
    """
    macs, params = profile(
        model,
        inputs=(
            torch.randn(*input_shape).to(next(model.parameters()).device),
        ),
    )
    return macs, params


def save_macs_and_parameters(macs, params, save_path):
    """将 macs 和参数量写入 save_path 目录下的报告文件。

    报告就写在传入目录里，不再写到它的上一级。原实现写的是
    ``save_path/../macs_and_parameters.txt``：当时 save_path 指模型目录，
    上一级是"同一模型所有实验共用"的位置。现在 save_path 是每次实验独立的
    时间戳目录，沿用旧写法会让第二次实验覆盖第一次的 MACs 记录——做 A/B
    对比时模型复杂度这一列就没法回溯了。

    目录不存在时自动创建（与 results.txt 的行为一致）。
    """
    os.makedirs(save_path, exist_ok=True)
    report_path = os.path.join(save_path, "macs_and_parameters.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"MACs = {macs / 1000**3:.4f} G\n")
        handle.write(f"Params = {params / 1000**2:.4f} M\n")
    return report_path
