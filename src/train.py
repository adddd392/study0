import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import Mini_ImageNet_Datasets
from models import get_model
import os
import argparse
from tqdm import tqdm
import torch_directml
import matplotlib.pyplot as plt
import time
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Resnet-50")
parser.add_argument("--data_root", type=str, default=r"..\data\Mini-ImageNet-Dataset")
parser.add_argument("--use_train_csv", type=bool, default=True)
parser.add_argument("--train_csv", type=str, default=r"..\data\Mini-ImageNet-Dataset\train.csv")
parser.add_argument("--use_val_csv", type=bool, default=True)
parser.add_argument("--val_csv", type=str, default=r"..\data\Mini-ImageNet-Dataset\val.csv")
parser.add_argument("--use_test_csv", type=bool, default=True)
parser.add_argument("--test_csv", type=str, default=r"..\data\Mini-ImageNet-Dataset\test.csv")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--device", type=str, default="dml")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--criterion", type=str, default="CrossEntropyLoss")
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--save_model", type=bool, default=True)
parser.add_argument("--save_path", type=str, default=os.path.join(r"..\saved_models",parser.parse_args().model_name,str(int(time.time()))))
args = parser.parse_args()
class trainer:
    def __init__(self,args):
        self.args=args
        self.device_init()
        self.model_init()
        self.criterion_init()
        self.optimizer_init()
        self.dataset_init()
    
    
    def dataset_init(self):
        if self.args.use_train_csv:
            self.train_dataset = Mini_ImageNet_Datasets(csv_file=self.args.train_csv, data_root=os.path.join(self.args.data_root, "train"))
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        if self.args.use_val_csv:
            self.val_dataset = Mini_ImageNet_Datasets(csv_file=self.args.val_csv, data_root=os.path.join(self.args.data_root, "val"))
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False)
        if self.args.use_test_csv:
            self.test_dataset = Mini_ImageNet_Datasets(csv_file=self.args.test_csv, data_root=os.path.join(self.args.data_root, "test"))
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False)

    def optimizer_init(self):
        if self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        else:
            raise ValueError("Wrong optimizer name")

    def device_init(self):
        if self.args.device == "dml":
            if not torch_directml.is_available():
                raise ValueError("DirectML is not available.")
            self.device = torch_directml.device()
        
    def model_init(self):
        self.model = get_model(self.args.model_name)
        self.model.to(self.device)
    
    def criterion_init(self):
        if self.args.criterion == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif self.args.criterion == "label_smoothing":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            raise ValueError("Wrong criterion name")

    def train_one_epoch(self):
        self.model.train()
        epoch_loss=0.0
        Top1_correct=0
        Top5_correct=0
        total=0
        for images,labels in tqdm(self.train_loader):
            images=images.to(self.device)
            labels=labels.to(self.device)
            self.optimizer.zero_grad()
            outputs=self.model(images)
            loss=self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss+=loss.item()*images.size(0)
            _,Top1_predicted=torch.max(outputs,1)
            total+=labels.size(0)
            Top1_correct+=(Top1_predicted==labels).sum().item()
            
            _,Top5_predicted=torch.topk(outputs,5,dim=1)
            Top5_correct+=(Top5_predicted.eq(labels.view(-1,1))).sum().item()
        epoch_Top1_acc=100.*Top1_correct/total
        epoch_Top5_acc=100.*Top5_correct/total
        epoch_loss/=total
        print(f"Epoch Loss: {epoch_loss:.4f}, Top-1 Accuracy: {epoch_Top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {epoch_Top5_acc:.2f}%")
        return epoch_loss,epoch_Top1_acc,epoch_Top5_acc

    def validate(self):
        self.model.eval()
        val_loss=0.0
        Top1_correct=0
        Top5_correct=0
        total=0
        with torch.no_grad():
            for images,labels in tqdm(self.val_loader):
                images=images.to(self.device)
                labels=labels.to(self.device)
                outputs=self.model(images)
                loss=self.criterion(outputs,labels)
                val_loss+=loss.item()*images.size(0)
                _,Top1_predicted=torch.max(outputs,1)
                total+=labels.size(0)
                Top1_correct+=(Top1_predicted==labels).sum().item()
                
                _,Top5_predicted=torch.topk(outputs,5,dim=1)
                Top5_correct+=(Top5_predicted.eq(labels.view(-1,1))).sum().item()
        val_Top1_acc=100.*Top1_correct/total
        val_Top5_acc=100.*Top5_correct/total
        val_loss/=total
        print(f"Validation Loss: {val_loss:.4f}, Top-1 Accuracy: {val_Top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {val_Top5_acc:.2f}%")
        return val_loss,val_Top1_acc,val_Top5_acc

    def test(self):
        self.model.eval()
        test_loss=0.0
        Top1_correct=0
        Top5_correct=0
        total=0
        with torch.no_grad():
            for images,labels in tqdm(self.test_loader):
                images=images.to(self.device)
                labels=labels.to(self.device)
                outputs=self.model(images)
                loss=self.criterion(outputs,labels)
                test_loss+=loss.item()*images.size(0)
                _,Top1_predicted=torch.max(outputs,1)
                total+=labels.size(0)
                Top1_correct+=(Top1_predicted==labels).sum().item()
                
                _,Top5_predicted=torch.topk(outputs,5,dim=1)
                Top5_correct+=(Top5_predicted.eq(labels.view(-1,1))).sum().item()
        test_Top1_acc=100.*Top1_correct/total
        test_Top5_acc=100.*Top5_correct/total
        test_loss/=total
        print(f"Test Loss: {test_loss:.4f}, Top-1 Accuracy: {test_Top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {test_Top5_acc:.2f}%")
        return test_loss,test_Top1_acc,test_Top5_acc

    def run_train(self):
        if not self.args.use_train_csv:
            print("Train CSV not provided.")
            return None,None,None,None
        training_losses=[]
        training_Top1_accuracies=[]
        training_Top5_accuracies=[]
        validation_losses=[]
        validation_Top1_accuracies=[]
        validation_Top5_accuracies=[]
        for epoch in range(self.args.num_epochs):
            print(f"Epoch{epoch+1}/{self.args.num_epochs}")
            train_loss,train_Top1_acc,train_Top5_acc=self.train_one_epoch()
            training_losses.append(train_loss)
            training_Top1_accuracies.append(train_Top1_acc)
            training_Top5_accuracies.append(train_Top5_acc)
            if self.args.use_val_csv:
                val_loss,val_Top1_acc,val_Top5_acc=self.validate()
                validation_losses.append(val_loss)
                validation_Top1_accuracies.append(val_Top1_acc)
                validation_Top5_accuracies.append(val_Top5_acc)
            if self.args.save_model:
                self.save_model(epoch+1)
        return training_losses,training_Top1_accuracies,training_Top5_accuracies,validation_losses,validation_Top1_accuracies,validation_Top5_accuracies

    def run_test(self):
        if not self.args.use_test_csv:
            print("Test CSV not provided.")
            return None,None,None
        test_loss,test_Top1_acc,test_Top5_acc=self.test()
        return test_loss,test_Top1_acc,test_Top5_acc

    def draw_loss_acc_curve(self,training_losses,training_Top1_accuracies,training_Top5_accuracies,validation_losses=None,validation_Top1_accuracies=None,validation_Top5_accuracies=None):
        epochs=range(1,self.args.num_epochs+1)
        plt.figure()
        plt.plot(epochs,training_losses,label="Training Loss")
        if validation_losses is not None:
            plt.plot(epochs,validation_losses,label="Validation Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        os.makedirs(os.path.join(self.args.save_path,"curves"), exist_ok=True)
        plt.savefig(os.path.join(self.args.save_path,"curves", "loss_curve.png"))
        plt.figure()
        plt.plot(epochs,training_Top1_accuracies,label="Training Top-1 Accuracy")
        plt.plot(epochs,training_Top5_accuracies,label="Training Top-5 Accuracy")
        if validation_Top1_accuracies is not None:
            plt.plot(epochs,validation_Top1_accuracies,label="Validation Top-1 Accuracy")
        if validation_Top5_accuracies is not None:
            plt.plot(epochs,validation_Top5_accuracies,label="Validation Top-5 Accuracy")
        plt.title("Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.show()
        os.makedirs(os.path.join(self.args.save_path,"curves"), exist_ok=True)
        plt.savefig(os.path.join(self.args.save_path,"curves", "accuracy_curve.png"))

    def run(self):
        training_losses,training_Top1_accuracies,training_Top5_accuracies,validation_losses,validation_Top1_accuracies,validation_Top5_accuracies=self.run_train()
        test_loss,test_Top1_acc,test_Top5_acc=self.run_test()
        self.draw_loss_acc_curve(training_losses=training_losses,
                                 training_Top1_accuracies=training_Top1_accuracies,
                                 training_Top5_accuracies=training_Top5_accuracies,
                                 validation_losses=validation_losses,
                                 validation_Top1_accuracies=validation_Top1_accuracies,
                                 validation_Top5_accuracies=validation_Top5_accuracies)
        return training_losses,training_Top1_accuracies,training_Top5_accuracies,validation_losses,validation_Top1_accuracies,validation_Top5_accuracies,test_loss,test_Top1_acc,test_Top5_acc


    def save_model(self,epoch):
        os.makedirs(os.path.join(self.args.save_path, "models"), exist_ok=True)
        save_path=os.path.join(self.args.save_path,"models" ,f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), save_path)



if __name__=="__main__":
    trainer_instance=trainer(args)
    trainer_instance.run()