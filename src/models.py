"""模型构建:按名字/类别数/输入通道数返回分类模型。

除类别数外，这里还要处理**输入通道数**。条码数据集是灰度图，输出单通道
张量，而 torchvision 的模型第一层是 3 通道。直接喂 1 通道会在第一个卷积处
报错，所以需要在构建阶段把第一层卷积换掉。

为什么不是"把灰度图复制成 3 通道"就完事:那样模型参数里仍有 3 份冗余权重，
第一层的表达能力被浪费；更重要的是，我们要的是**真正的单通道第一层**，
这样 MACs/参数量统计、量化、按通道剪枝都按 1 通道算，才和部署时的真实
计算量一致。

灰度化的权重要求和而不是取平均，原因见 _adapt_first_conv。
"""

import torch
import torch.nn as nn
import torchvision.models as models

MODEL_BUILDERS = {
    "Resnet-50": lambda n: models.resnet50(num_classes=n),
    "MobileNetV3-Large": lambda n: models.mobilenet_v3_large(num_classes=n),
    "ConvNeXt-Tiny": lambda n: models.convnext_tiny(num_classes=n),
}

# 各架构"第一个卷积层"在模型里的访问路径。
# ResNet 是 model.conv1；MobileNetV3 与 ConvNeXt 都把首层放在
# features[0][0] 里（ConvNormActivation 的第一个子模块）。
FIRST_CONV_PATHS = {
    "Resnet-50": ("conv1",),
    "MobileNetV3-Large": ("features", 0, 0),
    "ConvNeXt-Tiny": ("features", 0, 0),
}

# 支持替换的输入通道数。3 是原生通道数；1 用于灰度条码。
SUPPORTED_IN_CHANNELS = (1, 3)

# BatchNorm running 统计量的动量，统一成 PyTorch 默认值 0.1。
#
# 为什么需要这么一行：torchvision 的 MobileNetV3 用的是 TensorFlow 版配方
#     torchvision/models/mobilenetv3.py:154
#     norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
# 动量 0.01 意味着 running 统计量每步只向真实值挪 1%，初值(mean=0,var=1)的
# 残留比例是 0.99^步数，时间常数约 100 步。
#
# 本项目的训练预算远小于这个配方的假设：batch_size=32 时一轮只有 143 步。
# 实测（barcode_mobilenet_128 第 1 轮）：
#     瞬时 batch 方差恒为 0.079，而 running_var 从 1.0 只降到 0.299
#     即 eval 时的归一化参数里还残留 24% 的初始化值，46 层 BN 一路串联放大，
#     分类头退化：验证/测试准确率恒为 77.02%（= 全猜多数类 EAN13 的 754/979）
#     把统计量换成用训练集重算的正确值后，同一份权重是 94.38%（DML）/95.61%（CPU）
# 结论：权重是好的，坏的是评估时喂给网络的归一化参数。
#
# 改成 0.1 后，143 步的残留是 0.9^143≈3e-6，第 1 轮结束时统计量已完全收敛，
# 于是"第 1 轮就能拿到可信的验证指标"，模型选择（按 val loss 选最佳 epoch）
# 也不再建立在失真指标上。
#
# eps 故意保持 torchvision 的 0.001 不动：一次只动一个变量，便于归因。
BN_MOMENTUM = 0.1


def available_models() -> list:
    """返回可用模型名列表。"""
    return list(MODEL_BUILDERS.keys())


def first_conv_weight_key(model_name) -> str:
    """返回第一个卷积层权重在 state_dict 里的键。

    推理时必须从 checkpoint 反推输入通道数（灰度模型是 1、RGB 是 3），
    这个键就是单一来源。由 FIRST_CONV_PATHS 拼出来，不再单独维护一张表：

        Resnet-50         -> "conv1.weight"
        MobileNetV3-Large -> "features.0.0.weight"
    """
    path = _first_conv_path(model_name)
    return ".".join(str(step) for step in path) + ".weight"


def _walk(model, path):
    """按路径取出子模块，路径元素可以是属性名(str)或序列下标(int)。"""
    module = model
    for step in path:
        if isinstance(step, int):
            module = module[step]
        else:
            module = getattr(module, step)
    return module


def _set_submodule(model, path, new_module):
    """按路径替换子模块，父模块原地修改，返回被替换掉的旧模块。"""
    parent = _walk(model, path[:-1])
    last = path[-1]
    old = parent[last] if isinstance(last, int) else getattr(parent, last)
    if isinstance(last, int):
        parent[last] = new_module
    else:
        setattr(parent, last, new_module)
    return old


def _first_conv_path(model_name):
    """返回该架构第一个卷积层的访问路径，未知架构直接报错。"""
    if model_name not in FIRST_CONV_PATHS:
        raise ValueError(
            f"未登记首个卷积层路径: {model_name}，"
            f"可选：{sorted(FIRST_CONV_PATHS)}"
        )
    return FIRST_CONV_PATHS[model_name]


def _adapt_first_conv(model, model_name, in_channels):
    """把第一层卷积换成 in_channels 通道，其余超参完全保留。

    权重初始化规则(仅 1 通道):
        新权重 = 原权重在输入通道维上的**求和**。

    为什么是求和而不是取平均:设原卷积对像素 x 的输出为
    sum_c(w_c * x_c)。当输入是同一张灰度图复制三份时 x_0=x_1=x_2=x，
    输出等于 (w_0+w_1+w_2) * x。所以令 new_w = sum_c(w_c)，
    单通道输入 x 得到的结果与"三通道复制输入"完全一致 —— 预训练权重
    的响应尺度原封不动地保留下来，不需要再做任何缩放。

    参数:
        model: 已经构建好的模型（原地修改）。
        model_name: 架构名，用于定位第一层。
        in_channels: 目标输入通道数，必须是 SUPPORTED_IN_CHANNELS 之一。
    返回:
        同一个 model 对象。
    """
    if in_channels not in SUPPORTED_IN_CHANNELS:
        raise ValueError(
            f"in_channels 只支持 {SUPPORTED_IN_CHANNELS}，收到 {in_channels}"
        )
    path = _first_conv_path(model_name)
    conv = _walk(model, path)
    if conv.in_channels == in_channels:
        # 通道数本来就对，不动它，避免无意义地重置权重
        return model

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        # 沿输入通道维求和，见上面的推导
        new_conv.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    _set_submodule(model, path, new_conv)
    return model


def set_bn_momentum(model, momentum=BN_MOMENTUM):
    """把所有 BatchNorm 的 running 统计量动量统一设为 momentum。

    只改 momentum，不动 eps / affine / track_running_stats：
    归一化的形式不变，改变的只是"统计量跟随数据的速度"。

    MobileNetV3 出厂是 0.01（见 BN_MOMENTUM 的注释），改成 0.1 后
    running 统计量在十几步内就贴近真实分布，第 1 轮的 eval 指标即可信。
    ResNet 等本来就是 0.1 的模型调用本函数是空操作。

    返回模型本身，方便链式调用。
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum
    return model


def get_model(model_name, num_classes, in_channels=3) -> nn.Module:
    """返回指定架构、指定类别数与输入通道数的模型实例。

    参数:
        model_name: MODEL_BUILDERS 里的键。
        num_classes: 分类头输出维度。
        in_channels: 3 = 原生 RGB（默认，行为与重构前一致）；
                     1 = 灰度输入，第一层卷积会被替换。
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Wrong model_name,可选：{available_models()}")
    model = MODEL_BUILDERS[model_name](num_classes)
    # 必须在返回前统一动量：漏掉这一步，MobileNetV3 会带着 0.01 的动量
    # 训练，短预算下 eval 指标整体失真（见 BN_MOMENTUM 注释）
    set_bn_momentum(model, BN_MOMENTUM)
    return _adapt_first_conv(model, model_name, in_channels)
