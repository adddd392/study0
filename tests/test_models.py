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


def main() -> int:
    case_headers_and_params()
    case_grayscale_forward()
    case_grayscale_weight_is_sum()
    case_grayscale_matches_replicated_rgb()
    case_invalid_arguments()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
