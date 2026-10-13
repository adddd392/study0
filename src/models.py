import torch
import torch.nn as nn
import torchvision.models as models



def get_model(model_name,num_class):
    if model_name=="Resnet-50":
        return models.resnet50(num_classes=num_class)
    if model_name=="MobileNetV3-Large":
        return models.mobilenet_v3_large(num_classes=num_class)
    if model_name=="ConvNeXt-Tiny":
        return models.convnext_tiny(num_classes=num_class)
    raise ValueError("Wrong model_name")




if __name__=="__main__":
    model=get_model("Resnet-50",10)
    print(model)
    model=get_model("MobileNetV3-Large",10)
    print(model)
    model=get_model("ConvNeXt-Tiny",10)
    print(model)