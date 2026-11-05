"""Focal Loss 与类别权重生成的单元测试。

核心断言：gamma=0 且 alpha=None 时**必须**与标准交叉熵数值相等。
没有这条，无法区分"精度提升来自 Focal"还是"来自实现写错了"。
"""

import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from losses import FocalLoss, build_balanced_alpha  # noqa: E402


def _make_batch(seed=0):
    """构造一个固定的小批次，保证断言的确定性。"""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(16, 2, generator=generator)
    targets = torch.randint(0, 2, (16,), generator=generator)
    return logits, targets


def case_degenerates_to_cross_entropy():
    """gamma=0、alpha=None 时必须严格等于 nn.CrossEntropyLoss。"""
    logits, targets = _make_batch()
    focal = FocalLoss(gamma=0.0)
    cross_entropy = nn.CrossEntropyLoss()
    focal_value = focal(logits, targets)
    ce_value = cross_entropy(logits, targets)
    assert torch.allclose(focal_value, ce_value, atol=1e-6), (
        f"gamma=0 时 Focal Loss 应与交叉熵一致，"
        f"实际 {focal_value:.6f} vs {ce_value:.6f}"
    )


def case_hard_samples_dominate():
    """gamma>0 时，难样本（p_t 低）的损失占比必须高于 gamma=0 时。"""
    # 两个样本：第 0 个很容易（logit 差距大），第 1 个很难（logit 接近）
    logits = torch.tensor([[5.0, -5.0], [0.3, -0.3]])
    targets = torch.tensor([0, 0])
    plain = FocalLoss(gamma=0.0, reduction="none")(logits, targets)
    focal = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
    # 易分样本被压制的比例应该远大于难样本
    easy_suppression = focal[0] / plain[0]
    hard_suppression = focal[1] / plain[1]
    assert easy_suppression < hard_suppression, (
        "Focal Loss 应更多压制易分样本，"
        f"实际易分保留比 {easy_suppression:.4f} "
        f"vs 难分保留比 {hard_suppression:.4f}"
    )
    assert easy_suppression < 0.01, (
        f"p_t≈1 的样本应几乎不贡献损失，实际保留 {easy_suppression:.6f}"
    )


def case_alpha_weights_classes():
    """逐类 alpha 必须真正放大对应类别的损失。"""
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    targets = torch.tensor([0, 1])
    # C128 权重放大 2 倍
    weighted = FocalLoss(gamma=0.0, alpha=[2.0, 1.0],
                         reduction="none")(logits, targets)
    plain = FocalLoss(gamma=0.0, reduction="none")(logits, targets)
    assert torch.allclose(weighted[0], plain[0] * 2.0, atol=1e-6)
    assert torch.allclose(weighted[1], plain[1] * 1.0, atol=1e-6)


def case_scalar_alpha_supported():
    """alpha 传标量时所有样本乘同一个系数，不应报错。"""
    logits, targets = _make_batch()
    weighted = FocalLoss(gamma=1.0, alpha=2.0)(logits, targets)
    plain = FocalLoss(gamma=1.0)(logits, targets)
    assert torch.allclose(weighted, plain * 2.0, atol=1e-6)


def case_balanced_alpha_properties():
    """逆频率权重：少数类权重更大，且按样本数加权平均等于 1。"""
    counts = np.array([1053, 3516])
    alpha = build_balanced_alpha(counts)
    assert alpha[0] > alpha[1], (
        f"少数类 C128 的权重应更大，实际 {alpha.tolist()}"
    )
    # 归一化约束：sum(w * n) / sum(n) == 1
    weighted_mean = float((alpha * torch.as_tensor(counts)).sum()
                          / counts.sum())
    assert abs(weighted_mean - 1.0) < 1e-5, (
        f"权重加权平均应为 1，实际 {weighted_mean:.6f}"
    )


def case_balanced_alpha_power_zero():
    """power=0 时退化为均匀权重。"""
    alpha = build_balanced_alpha(np.array([10, 90]), power=0.0)
    assert torch.allclose(alpha, torch.ones(2), atol=1e-6), (
        f"power=0 应为均匀权重，实际 {alpha.tolist()}"
    )


def case_balanced_alpha_rejects_bad_input():
    """长度为 0 的类别、或与 num_classes 不一致时必须报错。"""
    for kwargs in (
        dict(class_counts=np.array([10, 0])),
        dict(class_counts=np.array([10, 20]), num_classes=3),
    ):
        try:
            build_balanced_alpha(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"非法输入 {kwargs} 应当报错，但没有")


def case_gradient_flows():
    """损失对 logits 必须可导，否则训练时不会更新任何参数。"""
    logits, targets = _make_batch()
    logits.requires_grad_(True)
    loss = FocalLoss(gamma=2.0, alpha=[0.77, 0.23])(logits, targets)
    loss.backward()
    assert logits.grad is not None, "损失没有产生梯度"
    assert torch.isfinite(logits.grad).all(), "梯度里出现了 nan/inf"


def main() -> int:
    cases = [
        ("退化性：gamma=0 等价交叉熵", case_degenerates_to_cross_entropy),
        ("难样本主导：易分样本被压制", case_hard_samples_dominate),
        ("类别加权：alpha 逐类生效", case_alpha_weights_classes),
        ("标量 alpha 可用", case_scalar_alpha_supported),
        ("逆频率权重：少数类更大且归一", case_balanced_alpha_properties),
        ("power=0 退化为均匀权重", case_balanced_alpha_power_zero),
        ("非法输入必须报错", case_balanced_alpha_rejects_bad_input),
        ("损失可导且梯度有限", case_gradient_flows),
    ]
    failed = 0
    for title, case in cases:
        try:
            case()
        except AssertionError as error:
            failed += 1
            print(f"[FAIL] {title}: {error}")
        else:
            print(f"[PASS] {title}")
    print(f"\n{len(cases) - failed}/{len(cases)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
