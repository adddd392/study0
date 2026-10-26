"""Bottleneck 残差单元的单元级验收测试。

范围只到单个 block，不涉及 ResNet 装配。这样在 ResNet 还没写完时
也能验证 Bottleneck 本身是否正确，把调试范围缩小一个数量级。

契约::

    Bottleneck(inplanes, planes, stride=1, downsample=None)
    Bottleneck.expansion == 4           # 类属性

    - inplanes  输入通道
    - planes    中间层通道；输出通道为 planes * expansion
    - stride    只作用于 3x3 那个卷积
    - downsample 旁路对齐分支，需要时由调用方（_make_layer）传入；
                 不传时旁路直接使用输入 x

    forward::

        main = bn3(conv3(relu(bn2(conv2(relu(bn1(conv1(x))))))))
        out  = relu(main + identity)
        identity = downsample(x) 或 x

覆盖 ResNet-50 四种真实配置：

    layer1 首块 / layer1 后续 / layer2 首块 / layer3 首块 / layer4 首块

注意 layer1 首块虽然 stride=1，但通道 64 -> 256 变化，同样需要 downsample。
若某个 block 的 expansion 为 1，此处两个条件都不成立，就无需 downsample。

运行::

    python tests/test_bottleneck.py
"""

import os
import sys

import torch
import torch.nn as nn
import torchvision.models.resnet as tvr

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import resnet  # noqa: E402

# (说明, inplanes, planes, stride, 是否需要 downsample, 输入边长)
CONFIGS = [
    ("layer1 首块", 64, 64, 1, True, 56),
    ("layer1 后续", 256, 64, 1, False, 56),
    ("layer2 首块", 256, 128, 2, True, 56),
    ("layer3 首块", 512, 256, 2, True, 28),
    ("layer4 首块", 1024, 512, 2, True, 14),
]


def make_downsample(inplanes, planes, stride):
    """构造旁路对齐分支：1x1 卷积 + BN，无 ReLU。"""
    return nn.Sequential(
        nn.Conv2d(
            inplanes, planes * 4, kernel_size=1, stride=stride, bias=False
        ),
        nn.BatchNorm2d(planes * 4),
    )


def build_pair(inplanes, planes, stride, use_downsample):
    """同时构造我们的实现与 torchvision 的参考实现，结构完全对称。"""

    def ds():
        """每次调用返回一个新的旁路分支，避免两边共享同一对象。"""
        if not use_downsample:
            return None
        return make_downsample(inplanes, planes, stride)

    ours = resnet.Bottleneck(
        inplanes, planes, stride=stride, downsample=ds()
    )
    ref = tvr.Bottleneck(inplanes, planes, stride=stride, downsample=ds())
    return ours, ref


def case_expansion() -> None:
    """expansion 必须是类属性，_make_layer 靠它计算输出通道。"""
    assert getattr(resnet.Bottleneck, "expansion", None) == 4, (
        "Bottleneck.expansion 必须是 4 且定义在类上"
    )
    print("expansion = 4 (类属性): OK")


def case_state_dict_and_params() -> None:
    """state_dict 键名与形状、参数量必须与 torchvision 一致。"""
    for title, inplanes, planes, stride, use_ds, _ in CONFIGS:
        ours, ref = build_pair(inplanes, planes, stride, use_ds)

        ours_keys = {k: tuple(v.shape) for k, v in ours.state_dict().items()}
        ref_keys = {k: tuple(v.shape) for k, v in ref.state_dict().items()}

        missing = sorted(set(ref_keys) - set(ours_keys))
        extra = sorted(set(ours_keys) - set(ref_keys))
        shape_diff = sorted(
            k
            for k in ref_keys
            if k in ours_keys and ours_keys[k] != ref_keys[k]
        )
        assert not missing, f"[{title}] 缺少键: {missing}"
        assert not extra, f"[{title}] 多余键: {extra}"
        assert not shape_diff, f"[{title}] 形状不对: {shape_diff}"

        # 没有 downsample 时不应出现 downsample.* 键
        has_ds_key = any(k.startswith("downsample.") for k in ours_keys)
        assert has_ds_key == use_ds, (
            f"[{title}] downsample 键存在性与预期不符（期望 {use_ds}）"
        )

        n_ours = sum(p.numel() for p in ours.parameters())
        n_ref = sum(p.numel() for p in ref.parameters())
        assert n_ours == n_ref, f"[{title}] 参数量 {n_ours} != {n_ref}"
    print(f"state_dict 键名/形状/参数量 ({len(CONFIGS)} 种配置): OK")


def case_forward_equivalence() -> None:
    """搬入权重后前向输出必须逐元素一致。"""
    for title, inplanes, planes, stride, use_ds, hw in CONFIGS:
        ours, ref = build_pair(inplanes, planes, stride, use_ds)
        ours.eval()
        ref.eval()
        ours.load_state_dict(ref.state_dict())

        torch.manual_seed(0)
        x = torch.randn(2, inplanes, hw, hw)
        with torch.no_grad():
            got, want = ours(x), ref(x)

        assert got.shape == want.shape, (
            f"[{title}] 输出形状 {tuple(got.shape)} != {tuple(want.shape)}"
        )
        diff = (got - want).abs().max().item()
        assert diff < 1e-5, (
            f"[{title}] 输出不一致（最大绝对差 {diff:.3e}）"
        )
    print(f"前向数值与 torchvision 一致 ({len(CONFIGS)} 种配置): OK")


def case_add_before_relu() -> None:
    """相加必须在 ReLU 之前：out = relu(main + identity)。"""
    ours, _ = build_pair(256, 64, 1, False)
    ours.eval()

    x = torch.randn(2, 256, 8, 8)
    with torch.no_grad():
        got = ours(x)
        main = torch.relu(ours.bn1(ours.conv1(x)))
        main = torch.relu(ours.bn2(ours.conv2(main)))
        main = ours.bn3(ours.conv3(main))
        want = torch.relu(main + x)

    diff = (got - want).abs().max().item()
    assert diff < 1e-5, (
        f"叠加顺序不对（最大绝对差 {diff:.3e}）。"
        "应为 relu(main + identity)，而不是 relu(main) + identity"
    )
    print("相加在 ReLU 之前: OK")


def case_identity_path() -> None:
    """旁路必须真的连通，且不被 ReLU 截断。

    手法：把 bn3 的 weight 与 bias 全部置 0，主路输出恒为 0，
    于是 out 应当等于 relu(identity)。取 identity 全为正数，
    则 out 必须等于输入本身，且 dx 的梯度必须恰好为 1。
    """
    block = resnet.Bottleneck(256, 64)  # 无 downsample，旁路即 x
    with torch.no_grad():
        block.bn3.weight.zero_()
        block.bn3.bias.zero_()
    block.eval()

    x = torch.rand(2, 256, 8, 8) + 1.0  # 全正，避开 ReLU 截断
    with torch.no_grad():
        out = block(x)

    diff = (out - x).abs().max().item()
    assert diff < 1e-6, (
        f"主路输出为 0 时，out 应等于输入（最大差 {diff:.3e}）。"
        "检查旁路是否真的接上、以及相加是否发生在 ReLU 之前"
    )

    x2 = (torch.rand(2, 256, 8, 8) + 1.0).requires_grad_(True)
    block(x2).sum().backward()
    grad_diff = (x2.grad - torch.ones_like(x2)).abs().max().item()
    assert grad_diff < 1e-6, (
        f"主路被置零后，输入梯度应恒为 1（实际最大偏差 {grad_diff:.3e}）。"
        "说明恒等旁路的梯度没有直通"
    )
    print("恒等旁路连通且梯度直通: OK")


def main() -> int:
    case_expansion()
    case_state_dict_and_params()
    case_forward_equivalence()
    case_add_before_relu()
    case_identity_path()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
