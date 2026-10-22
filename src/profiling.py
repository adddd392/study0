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
    """将 macs 和参数量写入报告文件。"""
    report_path = os.path.join(
        save_path, "..", "macs_and_parameters.txt"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"MACs = {macs / 1000**3:.4f} G\n")
        handle.write(f"Params = {params / 1000**2:.4f} M\n")
    return report_path
