import torch.nn as nn
import torchvision.models as models

MODEL_BUILDERS = {
    "Resnet-50": lambda n: models.resnet50(num_classes=n),
    "MobileNetV3-Large": lambda n: models.mobilenet_v3_large(num_classes=n),
    "ConvNeXt-Tiny": lambda n: models.convnext_tiny(num_classes=n),
}


def available_models() -> list[str]:
    """返回可用模型名列表。"""
    return list(MODEL_BUILDERS.keys())


def get_model(model_name, num_classes) -> nn.Module:
    """根据模型名和类别数返回对应分类头数量的模型实例。"""
    if model_name in MODEL_BUILDERS:
        model = MODEL_BUILDERS[model_name](num_classes)
        return model
    raise ValueError(f"Wrong model_name,可选：{available_models()}")
