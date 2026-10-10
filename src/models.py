import torch
import torch.nn as nn
import torchvision.models as models



def get_model(model_name):
    if model_name=="Resnet-50":
        return models.resnet50()
    if model_name=="MobileNetV3-Large":
        return models.mobilenet_v3_large()
    if model_name=="ConvNeXt-Tiny":
        return models.convnext_tiny()
    raise("Wrong model_name")


if __name__=="__main__":
    model=get_model("Resnet-50")
    print(model)
    model=get_model("MobileNetV3-Large")
    print(model)
    model=get_model("ConvNeXt-Tiny")
    print(model)