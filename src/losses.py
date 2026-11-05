"""损失函数集合。

目前只有 Focal Loss，用于处理条码数据集的类别不平衡
（C128 : EAN13 ≈ 1 : 3.3，而 C128 恰好是模型漏判最多的类）。

为什么需要它：标准交叉熵对所有样本一视同仁，训练后期大量"一眼就能分对"
的样本会主导梯度，少数类的错误反而得不到足够的纠正压力。Focal Loss 用
(1 - p_t)^gamma 把易分样本的损失压下去，让梯度集中在难样本上。

参考: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class FocalLoss(nn.Module):
    """Focal Loss：FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)。

    两个因子各管一件事：

        (1 - p_t)^gamma  调制因子。p_t 越接近 1（越容易分对），
                         (1-p_t)^gamma 越接近 0，该样本贡献的损失被压掉。
                         模型已经学会的样本不再贡献梯度，"难样本"权重上升。
        alpha_t          类别权重。第 t 个样本按其真实类别取权重，
                         用来补偿类别不平衡（少数类给更大权重）。

    两个参数同时为"默认"时（gamma=0、alpha=None）严格退化为交叉熵，
    这一点由单元测试保证 —— 否则无法判断增益来自 Focal 还是实现差异。
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """初始化。

        参数:
            gamma: 调制指数，>=0。0 表示不做调制（退化为加权交叉熵）。
                   原论文在检测任务上最优值是 2，这里默认沿用。
            alpha: 类别权重。可以是 None（不加权）、浮点数（作用于正类）、
                   或长度为类别数的序列/np 数组（逐类权重）。
            reduction: "mean" / "sum" / "none"，语义与 torch 的损失函数一致。
        """
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma 必须非负")
        self.gamma = float(gamma)
        self.reduction = reduction
        # 注册成 buffer：跟着 model.to(device) 走，不用手动搬设备。
        # 注意 register_buffer 的默认值必须能是 None，这里用 None 表示"不加权"。
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer(
                "alpha", torch.as_tensor(alpha, dtype=torch.float32)
            )

    def forward(self, inputs, targets):
        """计算损失。

        参数:
            inputs: (N, C) 的 logits（未经 softmax）。
            targets: (N,) 的整型类别索引。
        返回:
            标量损失（reduction="none" 时返回 (N,)）。
        """
        # log_softmax 比 softmax 再 log 数值更稳：C=2 时也不会出现 log(0)
        log_probability = functional.log_softmax(inputs, dim=1)
        # 取出每个样本"真实类别"对应的 log p_t
        log_p_t = log_probability.gather(
            dim=1, index=targets.view(-1, 1)
        ).squeeze(1)
        # p_t 直接由 log p_t 取指数得到，省一次 gather
        p_t = log_p_t.exp()

        # 十字交叉熵本体加调制因子：p_t 大则 (1-p_t) 小，损失被压制
        loss = -((1.0 - p_t) ** self.gamma) * log_p_t

        # 类别权重：按真实类别索引取 alpha_t
        if self.alpha is not None:
            if self.alpha.dim() == 0:
                # alpha 传的是标量：所有样本同一个权重
                loss = loss * self.alpha
            else:
                loss = loss * self.alpha[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_balanced_alpha(class_counts, num_classes=None, power=1.0):
    """按类别频率的倒数生成权重，用于 Focal Loss 的 alpha 或加权交叉熵。

    公式：w_c = (1 / n_c)^power，再归一化到"按样本数加权平均等于 1"，
    这样损失的整体尺度与不加权时可比，学习率不必重新调。

    参数:
        class_counts: 各类别样本数，序列。
        num_classes: 类别总数，None 时由 class_counts 长度推断。
        power: 1.0 表示完全按逆频率（标准 balanced 权重）；
               0.0 表示不加权；0.5 相当于"平方根平衡"，对极端不平衡更温和。
    返回:
        torch.Tensor，dtype=float32，长度 num_classes。
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    if num_classes is not None and len(counts) != num_classes:
        raise ValueError(
            f"class_counts 长度 {len(counts)} 与 num_classes "
            f"{num_classes} 不一致"
        )
    if np.any(counts <= 0):
        raise ValueError("每个类别的样本数都必须大于 0")

    weights = np.power(1.0 / counts, power)
    # 归一化约束：sum(w_c * n_c) / N == 1，保证整体损失量级不变
    weights = weights / (weights * counts).sum() * counts.sum()
    return torch.as_tensor(weights, dtype=torch.float32)
