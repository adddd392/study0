"""手写 ResNet-50 的验收测试。

契约::

    import resnet
    net = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1000)

    - 构造签名 ResNet(block, layers, num_classes=1000)；
    - 顶层属性 conv1 / bn1 / layer1..4 / avgpool / fc；
    - Bottleneck 单元属性 conv1(1x1 降维) / bn1 / conv2(3x3) / bn2 /
      conv3(1x1 升维) / bn3 / downsample，类属性 expansion = 4；
    - state_dict 的键名与形状必须与 torchvision.models.resnet50 完全一致，
      从而可以直接 load_state_dict 官方 ImageNet 预训练权重。

验证策略（逐级加强，出错时便于定位）::

    1. 键名 + 形状对齐      -> 抓命名、遗漏/多余模块、Sequential 写法
    2. 参数量对齐           -> 抓多一个少一个层
    3. 逐层中间输出对齐     -> 定位到 layer1/2/3/4 中的哪一层开始偏
    4. 最终输出逐元素对齐   -> 抓算子顺序错（先 ReLU 再加）、stride 位置错

第 3 步是本文件的重点：直接比最终输出，只能知道"错了"；
逐层比才知道"从哪开始错"。

运行::

    python tests/test_resnet.py
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as tv

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import resnet  # noqa: E402

LAYERS = [3, 4, 6, 3]  # ResNet-50

# ResNet-50 在 1000 类下的参数量（torchvision 实测）。
EXPECTED_PARAMS_1000 = 25_557_032

# fc 每类对应的参数：2048 个权重 + 1 个偏置。
FC_PARAMS_PER_CLASS = 2048 + 1


def build_ours(num_classes=1000):
    """构造我们的 ResNet-50 实现。"""
    return resnet.ResNet(resnet.Bottleneck, LAYERS, num_classes=num_classes)


def build_reference():
    """构造 torchvision 的 resnet50（weights=None，随机初始化，不联网）。"""
    return tv.resnet50(weights=None, num_classes=1000)


def param_items(model):
    """把 state_dict 转成 {键名: 形状元组}，便于比较结构。"""
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def case_structure() -> None:
    """键名与形状必须与 torchvision 完全一致。"""
    ours = param_items(build_ours())
    ref = param_items(build_reference())

    missing = sorted(set(ref) - set(ours))
    extra = sorted(set(ours) - set(ref))
    shape_diff = sorted(
        k for k in ref if k in ours and ours[k] != ref[k]
    )

    assert not missing, (
        "实现缺少这些键（对照 torchvision）：\n    "
        + "\n    ".join(f"{k} {ref[k]}" for k in missing[:15])
    )
    assert not extra, (
        "实现多出这些键：\n    "
        + "\n    ".join(f"{k} {ours[k]}" for k in extra[:15])
    )
    assert not shape_diff, (
        "这些键的形状不对：\n    "
        + "\n    ".join(
            f"{k}: 期望 {ref[k]}，实际 {ours[k]}" for k in shape_diff[:15]
        )
    )
    print(f"结构对齐 ({len(ref)} 个键名与形状): OK")


def case_parameter_count() -> None:
    """参数量必须精确匹配。"""
    ours = sum(p.numel() for p in build_ours().parameters())
    assert ours == EXPECTED_PARAMS_1000, (
        f"1000 类参数量期望 {EXPECTED_PARAMS_1000}，实际 {ours}"
        f"（相差 {ours - EXPECTED_PARAMS_1000:+d}）"
    )

    # 100 类是本项目的实际配置：只替换 fc 的参数
    expected_100 = (
        EXPECTED_PARAMS_1000
        - 1000 * FC_PARAMS_PER_CLASS
        + 100 * FC_PARAMS_PER_CLASS
    )
    ours_100 = sum(p.numel() for p in build_ours(100).parameters())
    assert ours_100 == expected_100, (
        f"100 类参数量期望 {expected_100}，实际 {ours_100}"
    )
    print(f"参数量 (1000 类 {EXPECTED_PARAMS_1000:,}，100 类 {expected_100:,}): OK")


def case_init_weights() -> None:
    """权重初始化必须按 torchvision 的方案。

    - 卷积：kaiming_normal_，mode="fan_out"，nonlinearity="relu"
      即权重标准差 = sqrt(2 / fan_out)，fan_out = 输出通道数 × 核高 × 核宽。
    - BatchNorm：weight 恒为 1，bias 恒为 0。

    PyTorch 对 nn.Conv2d 的默认初始化是 kaiming_uniform_(a=sqrt(5))，
    与上式差异显著（conv1 实测 std 约 0.047 与 0.025），
    因此本用例能明确区分"写了 _init_weights 并调用"与"没写"。

    注意：前向数值对齐用例会搬入 torchvision 权重，覆盖掉初始化，
    所以初始化必须单独验证，否则永远不会被发现遗漏。
    """
    net = build_ours()

    # 批归一化：weight 恰为 1，bias 恰为 0
    bn_count = 0
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            assert torch.allclose(
                module.weight, torch.ones_like(module.weight)
            ), f"{name}.weight 应全部为 1"
            assert torch.allclose(
                module.bias, torch.zeros_like(module.bias)
            ), f"{name}.bias 应全部为 0"
            bn_count += 1

    # 卷积：std 应等于 sqrt(2 / fan_out)
    conv_count = 0
    for name, module in net.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        out_channels, _, kh, kw = module.weight.shape
        fan_out = out_channels * kh * kw
        expected = math.sqrt(2.0 / fan_out)
        actual = module.weight.std().item()
        rel = abs(actual - expected) / expected
        assert rel < 0.10, (
            f"{name}: 权重 std {actual:.5f}，"
            f"kaiming_normal(fan_out) 期望 {expected:.5f}，相差 {rel:.1%}。"
            "检查是否实现了 _init_weights() 并在 __init__ 末尾调用"
        )
        conv_count += 1

    assert conv_count > 0 and bn_count > 0, "没有扫到卷积/BN，检查模块命名"
    print(
        f"权重初始化 (kaiming_normal fan_out {conv_count} 个卷积，"
        f"{bn_count} 个 BN): OK"
    )


def case_block_wiring() -> None:
    """检查 Bottleneck 与 _make_layer 的关键接线，这些位置出错不会报错但结果错。"""
    net = build_ours()

    # expansion 必须定义在类上，_make_layer 需要用它计算输出通道
    assert getattr(resnet.Bottleneck, "expansion", None) == 4, (
        "Bottleneck.expansion 必须是类属性且等于 4"
    )

    # stem：7x7 / stride 2 / padding 3 / 无 bias
    assert isinstance(net.conv1, nn.Conv2d)
    assert net.conv1.kernel_size == (7, 7), (
        f"conv1 核应为 7x7，实际 {net.conv1.kernel_size}"
    )
    assert net.conv1.stride == (2, 2), (
        f"conv1 stride 应为 2，实际 {net.conv1.stride}"
    )
    assert net.conv1.padding == (3, 3), (
        f"conv1 padding 应为 3，实际 {net.conv1.padding}"
    )
    assert net.conv1.bias is None, "conv1 后面接 BN，不应有 bias"

    # stem 的 maxpool 少了 padding=1 会让特征图变成 55x55，后面全错
    assert net.maxpool.kernel_size == 3
    assert net.maxpool.stride == 2
    assert net.maxpool.padding == 1, (
        f"maxpool padding 应为 1，实际 {net.maxpool.padding}"
    )

    # 分类头
    assert isinstance(net.avgpool, nn.AdaptiveAvgPool2d), (
        "avgpool 应用 AdaptiveAvgPool2d，才能适应任意输入尺寸"
    )
    assert net.fc.in_features == 2048, (
        f"fc 输入维度应为 2048，实际 {net.fc.in_features}"
    )

    # downsample：只在每个 layer 的第一个 block 出现。
    # ResNet-50 的四个 layer 首个 block 都把通道数放大 4 倍，
    # 所以即便 stride=1（layer1）也必须用 downsample 对齐通道。
    for name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(net, name)
        assert layer[0].downsample is not None, (
            f"{name} 第一个 block 的通道数变化（inplanes != planes*4），"
            "必须有 downsample；layer1 也不例外（64 -> 256，stride=1）"
        )
        assert layer[1].downsample is None, (
            f"{name} 只有第一个 block 需要 downsample"
        )

    # layer1 的 downsample 只改通道、不改尺寸；其余三层同时减半尺寸
    assert net.layer1[0].downsample[0].stride == (1, 1), (
        "layer1 的 downsample 只做通道对齐，stride 应为 1"
    )

    # downsample 必须是 Sequential(conv, bn)，且第二个元素是 BN 不是 ReLU
    ds = net.layer2[0].downsample
    assert isinstance(ds, nn.Sequential), (
        "downsample 必须是 nn.Sequential（键名 .0/.1）"
    )
    assert isinstance(ds[0], nn.Conv2d)
    assert isinstance(ds[1], nn.BatchNorm2d), (
        "downsample 的第二步应是 BatchNorm2d，不能加 ReLU"
    )
    assert ds[0].kernel_size == (1, 1), "downsample 的卷积应为 1x1"
    assert ds[0].stride == (2, 2), "downsample 的卷积 stride 应为 2"
    assert ds[0].out_channels == 512, (
        f"layer2 第一个 block 的 downsample 输出应为 128*4=512，实际 {ds[0].out_channels}"
    )

    # stride 必须放在 Bottleneck 的 3x3(conv2) 上，而不是第一个 1x1(conv1)
    blk = net.layer2[0]
    assert blk.conv1.kernel_size == (1, 1), "conv1 应为 1x1"
    assert blk.conv1.stride == (1, 1), (
        "stride 不应放在 conv1（ResNet v1.5 约定，torchvision 采用 v1.5）"
    )
    assert blk.conv2.kernel_size == (3, 3), "conv2 应为 3x3"
    assert blk.conv2.stride == (2, 2), (
        "Bottleneck 的 stride 必须放在 3x3(conv2) 上"
    )
    assert blk.conv3.kernel_size == (1, 1), "conv3 应为 1x1"

    # 同一 layer 内后续 block 一律 stride 1
    assert net.layer2[1].conv2.stride == (1, 1), (
        "_make_layer 中 stride 只作用于第一个 block"
    )
    print("Bottleneck 与 _make_layer 接线: OK")


def layer_outputs(model, x):
    """依次跑前向，收集 layer1..4 的输出，用于逐层比对。"""
    collected = {}
    handles = []
    for depth in (1, 2, 3, 4):
        name = f"layer{depth}"
        handles.append(
            getattr(model, name).register_forward_hook(
                lambda module, inputs, output, key=name: collected.__setitem__(
                    key, output.detach().clone()
                )
            )
        )
    with torch.no_grad():
        final = model(x)
    for handle in handles:
        handle.remove()
    return collected, final


def case_forward_equivalence() -> None:
    """搬入 torchvision 权重后，逐层中间输出与最终输出必须一致。"""
    ours = build_ours().eval()
    ref = build_reference().eval()

    try:
        ours.load_state_dict(ref.state_dict())
    except RuntimeError as error:
        raise AssertionError(
            "无法把 torchvision 权重加载进我们的实现，"
            f"说明 state_dict 结构不一致：\n{error}"
        )

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    ours_stages, ours_out = layer_outputs(ours, x)
    ref_stages, ref_out = layer_outputs(ref, x)

    # 逐层比对：第一个不一致的层就是问题所在
    hints = {
        "layer1": "第一个 block 若误加 downsample，或 stem 的 maxpool padding 错",
        "layer2": "第一个 block 的 stride / downsample 接线错",
        "layer3": "同上；或 _make_layer 用了 planes 而非 planes*expansion",
        "layer4": "同上",
    }
    for name in ("layer1", "layer2", "layer3", "layer4"):
        diff = (ours_stages[name] - ref_stages[name]).abs().max().item()
        assert diff < 1e-5, (
            f"{name} 输出与 torchvision 不一致（最大绝对差 {diff:.3e}）。"
            f"可能原因：{hints[name]}"
        )

    diff = (ours_out - ref_out).abs().max().item()
    assert diff < 1e-5, (
        f"最终输出不一致（最大绝对差 {diff:.3e}）。"
        "若中间层全对而最终输出错，重点检查分类头与最后一个 block 的 ReLU 位置"
    )
    print("前向数值对齐 (layer1..4 与最终输出): OK")


def case_residual_add_order() -> None:
    """残差相加必须在 ReLU 之前。

    单独成一个用例，是为了在"最终输出对不上"时给出比前向对齐更直接的
    诊断信息。手工复算只用到 state_dict 已经约束了的属性名
    （conv1..3 / bn1..3），不依赖 ReLU 叫什么名字。
    """
    net = build_ours().eval()
    # layer1[1]：inplanes=256、planes=64、stride=1，无 downsample，
    # 旁路可以直接用输入本身。layer1[0] 有 downsample（64->256），不能用。
    block = net.layer1[1]

    x = torch.randn(2, 256, 56, 56)
    with torch.no_grad():
        out = block(x)
        # 期望结构：relu -> relu -> 相加 -> relu
        main = torch.relu(block.bn1(block.conv1(x)))
        main = torch.relu(block.bn2(block.conv2(main)))
        main = block.bn3(block.conv3(main))
        expected = torch.relu(main + x)

    diff = (out - expected).abs().max().item()
    assert diff < 1e-5, (
        f"Bottleneck 叠加方式不对（最大绝对差 {diff:.3e}）。"
        "正确顺序是 out = relu(main + identity)，"
        "不是 relu(main) + identity"
    )
    print("残差相加在 ReLU 之前: OK")


def case_gradient_reaches_stem() -> None:
    """反向传播必须能到达最浅的 conv1，否则残差旁路没接上。"""
    net = build_ours(num_classes=100).train()
    out = net(torch.randn(2, 3, 128, 128))
    out.sum().backward()

    grad = net.conv1.weight.grad
    assert grad is not None, "conv1 没有梯度"
    assert grad.abs().sum().item() > 0, "conv1 梯度全为 0，浅层没收到信号"
    print("梯度可回流至 conv1: OK")


def main() -> int:
    case_structure()
    case_parameter_count()
    case_init_weights()
    case_block_wiring()
    case_forward_equivalence()
    case_residual_add_order()
    case_gradient_reaches_stem()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
