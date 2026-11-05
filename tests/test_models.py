"""models.py 的黄金测试。

锁定四件事：
    1. 分类头输出类数正确
    2. 主干网络总参数量不变（结构指纹）
    3. 灰度输入（in_channels=1）时第一层被正确替换，且前向可跑
    4. 灰度权重 = 原三通道权重求和 —— 等价于"把灰度图复制三份"喂原网络

参数量若变化，说明网络结构被改动过，必须停下来查。

运行::

    python tests/test_models.py
"""

import copy
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402

import models as models_module  # noqa: E402
from models import (  # noqa: E402
    FIRST_CONV_PATHS,
    _adapt_first_conv,
    _walk,
    available_models,
    get_model,
)

# 黄金数据：num_classes=10、三通道输入下测得，改动 models.py 后必须完全一致
EXPECTED_PARAMS = {
    "Resnet-50": 23528522,
    "MobileNetV3-Large": 4214842,
    # S16 变体只改一个卷积的 stride，参数量必须与原版**完全相同**
    "MobileNetV3-Large-S16": 4214842,
    "ConvNeXt-Tiny": 27827818,
}
NUM_CLASSES = 10

# 灰度测试用的输入尺寸：小一点，只验证通道与形状链路
GRAY_INPUT_SIZE = 64


def first_conv(model, name):
    """取模型的第一层卷积。"""
    return _walk(model, FIRST_CONV_PATHS[name])


def case_headers_and_params() -> None:
    """分类头类数正确，且总参数量与黄金数据一致。"""
    for name in available_models():
        model = get_model(name, NUM_CLASSES)
        header = model.fc if hasattr(model, "fc") else model.classifier[-1]

        assert header.out_features == NUM_CLASSES, (
            f"{name} 分类头不是 {NUM_CLASSES} 类"
        )

        total = sum(p.numel() for p in model.parameters())
        expected = EXPECTED_PARAMS[name]
        assert total == expected, (
            f"{name} 参数量变了: {total} != {expected}"
        )

        print(f"{name}: OK  params={total:,}  head={header.out_features}")


def case_grayscale_forward() -> None:
    """in_channels=1 时第一层被替换，参数量差值恰好是省掉的两个通道。"""
    for name in available_models():
        rgb = get_model(name, NUM_CLASSES)
        gray = get_model(name, NUM_CLASSES, in_channels=1)

        conv_rgb = first_conv(rgb, name)
        conv_gray = first_conv(gray, name)

        assert conv_rgb.in_channels == 3, f"{name} 原生第一层应为 3 通道"
        assert conv_gray.in_channels == 1, (
            f"{name} 灰度第一层仍是 {conv_gray.in_channels} 通道"
        )
        # 除输入通道外，其余超参必须原样保留
        assert conv_gray.out_channels == conv_rgb.out_channels
        assert conv_gray.kernel_size == conv_rgb.kernel_size
        assert conv_gray.stride == conv_rgb.stride
        assert conv_gray.padding == conv_rgb.padding

        # 参数量差值 = 去掉的两个输入通道 × 输出通道 × 卷积核面积
        out_c = conv_rgb.out_channels
        kernel_area = conv_rgb.kernel_size[0] * conv_rgb.kernel_size[1]
        removed = 2 * out_c * kernel_area
        params_rgb = sum(p.numel() for p in rgb.parameters())
        params_gray = sum(p.numel() for p in gray.parameters())
        assert params_rgb - params_gray == removed, (
            f"{name} 参数量差值 {params_rgb - params_gray} != {removed}"
        )

        # 前向能跑通
        gray.eval()
        with torch.no_grad():
            out = gray(torch.randn(2, 1, GRAY_INPUT_SIZE, GRAY_INPUT_SIZE))
        assert tuple(out.shape) == (2, NUM_CLASSES), (
            f"{name} 输出形状错: {tuple(out.shape)}"
        )
        print(
            f"{name} 灰度: OK  conv1.in={conv_gray.in_channels}  "
            f"params={params_gray:,}  (省 {removed:,})"
        )


def case_grayscale_weight_is_sum() -> None:
    """灰度权重必须等于原三通道权重逐点求和。"""
    for name in available_models():
        model = get_model(name, NUM_CLASSES)
        before = copy.deepcopy(first_conv(model, name).weight.detach())
        bias_before = first_conv(model, name).bias
        if bias_before is not None:
            bias_before = bias_before.detach().clone()

        _adapt_first_conv(model, name, 1)
        after = first_conv(model, name).weight.detach()

        expected = before.sum(dim=1, keepdim=True)
        assert torch.allclose(after, expected, atol=1e-6), (
            f"{name} 灰度权重不等于原权重求和（应求和而非求平均）"
        )
        if bias_before is not None:
            assert torch.allclose(
                first_conv(model, name).bias.detach(), bias_before
            ), f"{name} 替换第一层时不应改变 bias"
        print(f"{name} 灰度权重=原权重求和: OK")


def case_grayscale_matches_replicated_rgb() -> None:
    """核心性质：灰度输入的结果 == 把灰度复制三份喂给原模型。

    这条等式成立，灰度训练才和"RGB 复制"等价；否则第一层的响应尺度被
    改变，预训练权重的意义就没了。
    """
    for name in available_models():
        rgb = get_model(name, NUM_CLASSES)
        gray = copy.deepcopy(rgb)
        _adapt_first_conv(gray, name, 1)
        rgb.eval()
        gray.eval()

        # 一张灰度图，形状 (N, 1, H, W)
        x = torch.randn(2, 1, GRAY_INPUT_SIZE, GRAY_INPUT_SIZE)
        with torch.no_grad():
            out_gray = gray(x)
            # 同一张图复制成三通道喂给原模型
            out_rgb = rgb(x.repeat(1, 3, 1, 1))

        max_diff = float((out_gray - out_rgb).abs().max())
        assert max_diff < 1e-4, (
            f"{name} 灰度输入与原模型吃复制RGB的结果不一致，最大差 {max_diff}"
        )
        print(f"{name} 灰度 ≡ 复制RGB: OK  max_diff={max_diff:.2e}")


def case_invalid_arguments() -> None:
    """非法输入通道数与未知模型名必须报错。"""
    try:
        get_model("Resnet-50", NUM_CLASSES, in_channels=4)
    except ValueError as exc:
        assert "in_channels" in str(exc), f"报错信息不明确: {exc}"
    else:
        raise AssertionError("in_channels=4 应抛 ValueError")

    try:
        get_model("不存在的模型", NUM_CLASSES)
    except ValueError as exc:
        assert "model_name" in str(exc), f"报错信息不明确: {exc}"
    else:
        raise AssertionError("未知模型名应抛 ValueError")
    print("非法参数校验: OK")


def case_bn_momentum_unified() -> None:
    """所有模型的 BatchNorm 动量必须统一为 BN_MOMENTUM。

    这条是"评估指标失真"事故的防复发测试。torchvision 的 MobileNetV3
    出厂动量是 0.01（TensorFlow 配方），本项目的训练预算只有每轮 143 步，
    running 统计量根本收敛不了，于是：
        - 训练指标正常（train 模式用 batch 统计量，自洽）
        - 验证/测试指标恒为多数类比例（eval 模式用了错的 running 统计量）
    实测症状：连续 3 轮验证准确率一字不差 77.02%，检查点里的 running_var
    比用训练集重算的真实值大 4~22 倍。

    动量统一后，143 步的初值残留从 24% 降到 3e-6，第 1 轮指标即可信。
    """
    checked = []
    for name in available_models():
        model = get_model(name, NUM_CLASSES)
        momentums = {
            module.momentum
            for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        }
        if not momentums:
            # ConvNeXt 用 LayerNorm，没有 BN，天然不存在统计量滞后问题
            print(f"{name}: 无 BatchNorm（用 LayerNorm），跳过")
            continue
        assert momentums == {models_module.BN_MOMENTUM}, (
            f"{name} 的 BN 动量未统一: {sorted(momentums)}"
            f" != {{{models_module.BN_MOMENTUM}}}"
        )
        checked.append(name)
        print(f"{name}: BN 动量 {momentums.pop()} OK")
    assert checked, "没有任何模型含 BatchNorm，测试前提不成立"


def case_bn_momentum_helper() -> None:
    """set_bn_momentum 只改动量，不动 eps 与其它归一化参数。"""
    # 造一个"出厂动量 0.01"的 BN，模拟 MobileNetV3 的情况
    module = torch.nn.BatchNorm2d(
        8, eps=0.001, momentum=0.01
    )
    assert module.momentum == 0.01

    wrapper = torch.nn.Sequential(module)
    models_module.set_bn_momentum(wrapper, 0.1)

    assert module.momentum == 0.1, f"动量未改: {module.momentum}"
    assert module.eps == 0.001, f"eps 不该被改动: {module.eps}"
    assert module.affine is True, "affine 不该被改动"
    assert module.track_running_stats is True, "track_running_stats 不该被改动"
    print("set_bn_momentum: 只改动量 OK")


def case_backbone_stride_reduced() -> None:
    """S16 变体的累计步长必须是 16，原版是 32，且只改了一个 block。

    这条是"降低下采样"实验的结构指纹：如果哪天有人多改/少改一个步长，
    特征图尺寸就会变，精度对比立刻失去意义。
    """
    native = get_model("MobileNetV3-Large", NUM_CLASSES)
    reduced = get_model("MobileNetV3-Large-S16", NUM_CLASSES)

    assert models_module.backbone_stride(native) == 32, (
        f"原生 MobileNetV3 累计步长应为 32，"
        f"实际 {models_module.backbone_stride(native)}"
    )
    assert models_module.backbone_stride(reduced) == 16, (
        f"S16 变体累计步长应为 16，"
        f"实际 {models_module.backbone_stride(reduced)}"
    )

    # 只改最深的 block[13]，其余 block 保持原样
    changed = models_module.reduce_backbone_stride(
        get_model("MobileNetV3-Large", NUM_CLASSES), 16
    )
    assert changed == [13], f"S16 只应改 block[13] 的步长，实际改了 {changed}"
    print(f"累计步长: 原版 32 -> S16 16 OK（改动 block {changed}）")


def case_feature_map_doubles() -> None:
    """224 输入时，S16 的分类头前特征图必须是 14x14（原版 7x7）。

    同时验证 avgpool 之后仍能被分类头正确消费——步长改了但 head 不用动。
    """
    size = 224
    native = get_model("MobileNetV3-Large", NUM_CLASSES, in_channels=1).eval()
    reduced = get_model("MobileNetV3-Large-S16", NUM_CLASSES,
                        in_channels=1).eval()

    probe = torch.zeros(1, 1, size, size)
    with torch.no_grad():
        native_map = native.features(probe)
        reduced_map = reduced.features(probe)
        reduced_logits = reduced(probe)

    assert tuple(native_map.shape[2:]) == (7, 7), (
        f"原版特征图应为 7x7，实际 {tuple(native_map.shape[2:])}"
    )
    assert tuple(reduced_map.shape[2:]) == (14, 14), (
        f"S16 特征图应为 14x14，实际 {tuple(reduced_map.shape[2:])}"
    )
    assert tuple(reduced_logits.shape) == (1, NUM_CLASSES), (
        f"S16 输出形状错: {tuple(reduced_logits.shape)}"
    )
    print(
        f"特征图: 原版 {tuple(native_map.shape)} -> "
        f"S16 {tuple(reduced_map.shape)} OK"
    )


def case_reduce_stride_rejects_impossible_target() -> None:
    """目标步长比可取消的下采样还小，必须报错而不是静默返回。"""
    # MobileNetV3 最多只能取消 5 个步长 2（32 -> 1）
    model = get_model("MobileNetV3-Large", NUM_CLASSES)
    models_module.reduce_backbone_stride(model, 16)
    assert models_module.backbone_stride(model) == 16

    try:
        models_module.reduce_backbone_stride(model, 0)
    except ValueError as exc:
        assert "target_stride" in str(exc), f"报错信息不明确: {exc}"
    else:
        raise AssertionError("target_stride=0 应抛 ValueError")
    print("非法 target_stride 校验: OK")


def main() -> int:
    case_headers_and_params()
    case_grayscale_forward()
    case_grayscale_weight_is_sum()
    case_grayscale_matches_replicated_rgb()
    case_invalid_arguments()
    case_bn_momentum_unified()
    case_bn_momentum_helper()
    case_backbone_stride_reduced()
    case_feature_map_doubles()
    case_reduce_stride_rejects_impossible_target()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
