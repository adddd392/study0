from train import trainer
import time
import os
from types import SimpleNamespace
args={
    "model_name":"Resnet-50",
    "data_root":r"..\data\Mini-ImageNet-Dataset",
    "use_train_csv":True,
    "train_csv":r"..\data\Mini-ImageNet-Dataset\train.csv",
    "use_val_csv":True,
    "val_csv":r"..\data\Mini-ImageNet-Dataset\val.csv",
    "use_test_csv":True,
    "test_csv":r"..\data\Mini-ImageNet-Dataset\test.csv",
    "num_epochs":30,
    "device":"cuda",
    "batch_size":32,
    "learning_rate":0.001,
    "criterion":"CrossEntropyLoss",
    "optimizer":"Adam",
    "save_model":True,
    "save_path":os.path.join(r"..\saved_models",r"Resnet-50",str(int(time.time()))),
    "input_size":(224,224)
}
args = SimpleNamespace(**args)
if __name__ == '__main__':
    trainer = trainer(args)
    trainer.run()
    trainer.get_model_flops_and_parameters()
    trainer.latency_test()
    trainer.switch_device("cpu")
    trainer.latency_test()
    trainer.switch_device("cuda")
    trainer.switch_model("MobileNetV3-Large")
    trainer.run()
    trainer.get_model_flops_and_parameters()
    trainer.latency_test()
    trainer.switch_device("cpu")
    trainer.latency_test() 
    trainer.switch_device("cuda")
    trainer.switch_model("ConvNeXt-Tiny")
    trainer.run()
    trainer.get_model_flops_and_parameters()
    trainer.latency_test()
    trainer.switch_device("cpu")
    trainer.latency_test() 
    trainer.switch_device("cuda")
    trainer.switch_model("ConvNeXt-Tiny")
    trainer.switch_criterion("label_smoothing")
    trainer.run()
    