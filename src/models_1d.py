"""一维主干：把 2D 的 MobileNetV3 自动转换成 1D 版本。

为什么用"转换"而不是"另写一个 1D 网络"
--------------------------------------
本实验要回答的是"2D 布局贡献了多少精度"。如果 1D 侧换成完全不同的
网络（比如自研的 ProfileNet），那"精度差"就同时混进了**骨架差异**
和**输入表示差异**，结论不可用。

所以做法是：**同一份 2D 构建代码 → 逐层替换成 1D 对应物**。
替换后参数量几乎不变（4.204M → 4.134M，只差深度卷积核 k*k -> k），
训练协议完全一致，唯一被改动的变量就是输入张量本身。

层映射（一一对应，不留 2D 残留）
--------------------------------
    Conv2d(k, k)                -> Conv1d(k)
    BatchNorm2d                 -> BatchNorm1d（沿用 momentum=0.1）
    MaxPool2d                   -> MaxPool1d
    AdaptiveAvgPool2d           -> AdaptiveAvgPool1d
    SqueezeExcitation (2D)      -> SqueezeExcitation1d（torchvision 没有 1D 版）

`convert_to_1d` 返回替换计数，测试里用它做"是否还有 2D 残留"的断言。
"""

import torch.nn as nn

# 复用 2D 的构建器与访问工具，避免把 NAS 配置抄第二份（抄了就会漂移）
import models as models_2d


# 1D 架构名 -> 对应的 2D 架构名。
# 为什么要两个 S32/S16 变体：2D 那边我们做过"降主干步长"的实验，
# 1D 侧保留同样的两个档位，才能把"末端特征粒度"这个因素对齐。
ONE_D_SOURCES = {
    "MobileNetV3-Large-1D": "MobileNetV3-Large",
    "MobileNetV3-Large-S16-1D": "MobileNetV3-Large-S16",
}

# 第一个卷积层在模型里的访问路径（与 2D 的 features[0][0] 一致）
FIRST_CONV_PATH_1D = ("features", 0, 0)


class SqueezeExcitation1d(nn.Module):
    """torchvision SqueezeExcitation 的 1D 版本，数学上完全对应。

    SE 的作用：先全局池化把每个通道压成一个数（通道描述符），
    经一个小 MLP 学出每个通道的重要性权重，再乘回原特征。
    在 1D 里"空间维"只是一条 L 长的序列，池化变成 AdaptiveAvgPool1d，
    其余（两个 1x1 卷积 + 激活 + 缩放）完全不变。
    """

    def __init__(self, in_channels, squeeze_channels, scale_activation):
        super().__init__()
        # 全局池化：把 (N, C, L) 压成 (N, C, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 降维卷积，等价于 2D 版里的 fc1（1x1 卷积就是逐像素全连接）
        self.fc1 = nn.Conv1d(in_channels, squeeze_channels, 1)
        # 升维卷积，恢复到原通道数
        self.fc2 = nn.Conv1d(squeeze_channels, in_channels, 1)
        # 与 torchvision 一致：先 ReLU 再 Hardsigmoid
        self.activation = nn.ReLU()
        self.scale_activation = scale_activation

    def _scale(self, x):
        """算出每个通道的缩放系数，形状 (N, C, 1)，可直接广播相乘。"""
        s = self.avgpool(x)
        s = self.fc1(s)
        s = self.activation(s)
        s = self.fc2(s)
        return self.scale_activation(s)

    def forward(self, x):
        # 原地相乘会让 autograd 报错，所以写成 x * scale
        return x * self._scale(x)


def _to_1d_conv(conv):
    """Conv2d -> Conv1d，只取空间维的第一个元素。"""
    # kernel/stride/padding/dilation 在 2D 里是二元组，1D 只要第一个分量。
    # 注意：MobileNetV3 的下采样 stride 放在**深度卷积**上，
    # 这里逐层替换即可，不需要关心它在哪个子模块。
    kernel = conv.kernel_size[0]
    stride = conv.stride[0]
    padding = conv.padding[0]
    dilation = conv.dilation[0]
    return nn.Conv1d(
        conv.in_channels,
        conv.out_channels,
        kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )


def _to_1d_bn(bn):
    """BatchNorm2d -> BatchNorm1d，eps 与 momentum 原样保留。"""
    # momentum 必须保留：2D 侧已被统一成 0.1，这里若用默认值会产生
    # "同一个实验里两种动量"的隐蔽差异
    return nn.BatchNorm1d(
        bn.num_features,
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    )


def _to_1d_pool(pool):
    """MaxPool2d -> MaxPool1d。"""
    kernel = pool.kernel_size
    if isinstance(kernel, tuple):
        kernel = kernel[0]
    stride = pool.stride
    if isinstance(stride, tuple):
        stride = stride[0]
    padding = pool.padding
    if isinstance(padding, tuple):
        padding = padding[0]
    return nn.MaxPool1d(kernel, stride=stride, padding=padding)


def _to_1d_adaptive_pool(pool):
    """AdaptiveAvgPool2d -> AdaptiveAvgPool1d。"""
    out = pool.output_size
    if not isinstance(out, int):
        # 2D 的 output_size 可能是 (1, 1)；两个维度必须一致，否则说明
        # 这个池化不是"全局池化"，不能简单降成一维
        if out[0] != out[1]:
            raise ValueError("非对称自适应池化无法转成 1D: %r" % (out,))
        out = out[0]
    return nn.AdaptiveAvgPool1d(out)


def convert_to_1d(module):
    """原地把 2D 层替换成 1D 对应物，返回各类型的替换计数。

    递归遍历所有子模块；不带参数的容器（Sequential / InvertedResidual /
    ConvBNActivation）会继续向下找，直到把叶子层的 2D 版本全部换掉。

    返回:
        dict，键为 conv / bn / pool / se，值为替换次数。
        测试里用它断言"确实替换了"，并在转换后复查"是否还有 2D 残留"。
    """
    counts = {"conv": 0, "bn": 0, "pool": 0, "se": 0}
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, _to_1d_conv(child))
            counts["conv"] += 1
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, _to_1d_bn(child))
            counts["bn"] += 1
        elif isinstance(child, nn.MaxPool2d):
            setattr(module, name, _to_1d_pool(child))
            counts["pool"] += 1
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, _to_1d_adaptive_pool(child))
            counts["pool"] += 1
        elif type(child).__name__ == "SqueezeExcitation":
            # torchvision 的 SE 是 2D 专用，用自己写的 1D 版替换
            setattr(module, name, SqueezeExcitation1d(
                child.fc1.in_channels,
                child.fc1.out_channels,
                child.scale_activation,
            ))
            counts["se"] += 1
        else:
            # 容器模块：递归下去找叶子层
            sub = convert_to_1d(child)
            for key, value in sub.items():
                counts[key] += value
    return counts


def residual_2d_layers(model):
    """返回模型里残留的 2D 层类型名列表（转换后应为空）。"""
    bad = []
    for module in model.modules():
        name = type(module).__name__
        if name in ("Conv2d", "BatchNorm2d", "MaxPool2d",
                    "AdaptiveAvgPool2d", "SqueezeExcitation"):
            bad.append(name)
    return bad


def _replace_first_conv(model, in_channels):
    """把 1D 首层换成 in_channels 通道。

    这里不做权重搬运：1D 模型是从零训练的，首层通道数（2）与 2D 的
    三通道没有对应关系，硬搬权重只会引入无意义的初始化偏置。
    """
    conv = models_2d._walk(model, FIRST_CONV_PATH_1D)
    new_conv = nn.Conv1d(
        in_channels,
        conv.out_channels,
        conv.kernel_size[0],
        stride=conv.stride[0],
        padding=conv.padding[0],
        dilation=conv.dilation[0],
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    models_2d._set_submodule(model, FIRST_CONV_PATH_1D, new_conv)
    return model


def available_profile_models():
    """返回已登记的 1D 架构名列表。"""
    return sorted(ONE_D_SOURCES)


def get_profile_model(model_name, num_classes=2, in_channels=2):
    """返回指定 1D 架构的模型实例。

    参数:
        model_name: ONE_D_SOURCES 里的键。
        num_classes: 分类头输出维度。
        in_channels: 剖面通道数（2 = 平滑 + 高通），默认 2。

    实现顺序很重要：
        1. 用 2D 构建器建**三通道**模型（此时 BN 动量已被统一成 0.1）；
        2. 逐层替换成 1D；
        3. 最后替换首层通道数。
    先转换再换首层，是为了让首层替换复用同一套 Conv1d 参数解析逻辑。
    """
    if model_name not in ONE_D_SOURCES:
        raise ValueError(
            "未登记的 1D 架构: %s，可选: %s"
            % (model_name, available_profile_models())
        )
    source = ONE_D_SOURCES[model_name]
    # in_channels 传 3：先按原生 RGB 建，转换完再改首层
    model = models_2d.get_model(source, num_classes, in_channels=3)
    convert_to_1d(model)
    _replace_first_conv(model, in_channels)
    left = residual_2d_layers(model)
    if left:
        # 转换不彻底会让 1D 输入在某一层崩掉，必须当场暴露
        raise RuntimeError("转换后仍有 2D 层残留: %s" % sorted(set(left)))
    return model
