"""models_1d.py 的黄金测试。

锁定六件事：
    1. 转换彻底：模型里不再有任何 2D 层残留
    2. 参数总量不变（结构指纹）—— 4,133,506
    3. 前向可跑，输出 (N, num_classes)，末端特征长度符合预期（32 / 64）
    4. 首层输入通道数为剖面通道数（2）
    5. BN 动量被统一成 0.1（与 2D 基线同口径）
    6. SE 的 1D 版数学上正确（缩放系数落在 (0,1)，且与手算一致）

运行::

    python tests/test_profile_models.py
"""

import os
import sys

import torch
import torch.nn as nn

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from models_1d import (  # noqa: E402
    SqueezeExcitation1d,
    available_profile_models,
    convert_to_1d,
    get_profile_model,
    residual_2d_layers,
)

# 各 1D 架构的参数总量（结构指纹：变了就说明骨架被改过）
EXPECTED_PARAMS = {
    "MobileNetV3-Large-1D": 4133506,
    "MobileNetV3-Large-S16-1D": 4133506,
}

# 1024 长度输入下，分类头之前最后一个 block 的输出长度
EXPECTED_END_LENGTH = {
    "MobileNetV3-Large-1D": 32,     # 2D 版是 7x7，对应累计步长 32
    "MobileNetV3-Large-S16-1D": 64,  # 累计步长 16
}


def case_no_2d_residual() -> None:
    """两个 1D 架构转换后都不能有 2D 层残留。"""
    for name in available_profile_models():
        model = get_profile_model(name, num_classes=2, in_channels=2)
        left = residual_2d_layers(model)
        assert not left, "%s 仍有 2D 层残留: %s" % (name, sorted(set(left)))
        print("  %s: 无 2D 残留" % name)
    print("case_no_2d_residual: OK")


def case_param_counts() -> None:
    """参数总量必须等于黄金值（结构指纹）。"""
    for name in available_profile_models():
        model = get_profile_model(name, num_classes=2, in_channels=2)
        n_par = sum(p.numel() for p in model.parameters())
        expected = EXPECTED_PARAMS[name]
        assert n_par == expected, (
            "%s 参数量 %d != 黄金值 %d，骨架被改动过"
            % (name, n_par, expected)
        )
        print("  %s: 参数 %d" % (name, n_par))
    print("case_param_counts: OK")


def case_forward_and_end_length() -> None:
    """前向可跑，输出形状正确，末端特征长度符合累计步长。"""
    for name in available_profile_models():
        model = get_profile_model(name, num_classes=2, in_channels=2)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 2, 1024))
        assert out.shape == (2, 2), "%s 输出形状 %r" % (name, tuple(out.shape))
        # 逐层跑一遍，拿到分类头之前的长度
        feat = torch.zeros(1, 2, 1024)
        with torch.no_grad():
            for layer in model.features:
                feat = layer(feat)
        end = feat.shape[-1]
        assert end == EXPECTED_END_LENGTH[name], (
            "%s 末端长度 %d != %d" % (name, end, EXPECTED_END_LENGTH[name])
        )
        print("  %s: 输出 %s, 末端长度 %d"
              % (name, tuple(out.shape), end))
    print("case_forward_and_end_length: OK")


def case_other_lengths() -> None:
    """其他剖面长度也应可跑（L=512 / 2048 都要支持）。"""
    model = get_profile_model("MobileNetV3-Large-1D", num_classes=2)
    model.eval()
    for length in (512, 2048):
        with torch.no_grad():
            out = model(torch.zeros(1, 2, length))
        assert out.shape == (1, 2), "L=%d 输出形状不对" % length
    print("case_other_lengths: OK")


def case_first_conv_channels() -> None:
    """首层输入通道数应等于剖面通道数，且必须是 Conv1d。"""
    for channels in (1, 2, 4):
        model = get_profile_model("MobileNetV3-Large-1D", num_classes=2,
                                  in_channels=channels)
        conv = model.features[0][0]
        assert isinstance(conv, nn.Conv1d), "首层不是 Conv1d"
        assert conv.in_channels == channels, (
            "首层输入通道 %d != %d" % (conv.in_channels, channels)
        )
    print("case_first_conv_channels: OK")


def case_bn_momentum() -> None:
    """所有 BN 的动量必须是 0.1，与 2D 基线同口径。"""
    model = get_profile_model("MobileNetV3-Large-1D", num_classes=2)
    momenta = {m.momentum for m in model.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm)}
    assert momenta == {0.1}, "BN 动量不统一: %r" % (momenta,)
    print("case_bn_momentum: OK")


def case_se_1d_math() -> None:
    """SE 的 1D 版：缩放系数在 (0,1)，且与手算的公式一致。"""
    torch.manual_seed(0)
    se = SqueezeExcitation1d(8, 4, nn.Hardsigmoid())
    x = torch.randn(2, 8, 16)
    scale = se._scale(x)
    assert scale.shape == (2, 8, 1), "缩放系数形状 %r" % (tuple(scale.shape),)
    assert float(scale.min()) > 0.0 and float(scale.max()) < 1.0, (
        "Hardsigmoid 输出应严格落在 (0, 1)"
    )
    # 手算对照：pool -> fc1 -> relu -> fc2 -> hardsigmoid
    manual = torch.nn.functional.hardsigmoid(
        se.fc2(torch.relu(se.fc1(x.mean(dim=2, keepdim=True))))
    )
    assert torch.allclose(scale, manual, atol=1e-6), "SE 手算结果不一致"
    # 前向应是逐通道缩放
    assert torch.allclose(se(x), x * scale, atol=1e-6), "SE 前向不是逐通道缩放"
    print("case_se_1d_math: OK")


def case_convert_counts() -> None:
    """转换计数必须与 2D 骨架的层数吻合，且不重复替换。"""
    model = get_profile_model("MobileNetV3-Large-1D", num_classes=2)
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv1d))
    n_bn = sum(1 for m in model.modules()
               if isinstance(m, nn.BatchNorm1d))
    counts = convert_to_1d(model)          # 二次转换应全部为 0
    assert counts == {"conv": 0, "bn": 0, "pool": 0, "se": 0}, (
        "二次转换不应再替换任何层，实际 %r" % (counts,)
    )
    assert n_conv == 62, "Conv1d 数量 %d != 62" % n_conv
    assert n_bn == 46, "BatchNorm1d 数量 %d != 46" % n_bn
    print("case_convert_counts: OK")


def case_invalid_name() -> None:
    """未登记的架构名应抛 ValueError，且提示可选值。"""
    try:
        get_profile_model("Not-A-Model", num_classes=2)
    except ValueError as exc:
        assert "可选" in str(exc), "错误信息里应给出可选架构"
    else:
        raise AssertionError("非法架构名应抛 ValueError")
    print("case_invalid_name: OK")


def case_num_classes() -> None:
    """分类头输出维度应跟随 num_classes。"""
    for num_classes in (2, 10):
        model = get_profile_model("MobileNetV3-Large-1D",
                                  num_classes=num_classes)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 2, 1024))
        assert out.shape == (1, num_classes), "输出维度不对"
    print("case_num_classes: OK")


def main() -> int:
    case_no_2d_residual()
    case_param_counts()
    case_forward_and_end_length()
    case_other_lengths()
    case_first_conv_channels()
    case_bn_momentum()
    case_se_1d_math()
    case_convert_counts()
    case_invalid_name()
    case_num_classes()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
