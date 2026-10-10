import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
class Mini_ImageNet_Datasets(Dataset):
    def __init__(self,csv_file,data_root,inputsize=(224,224)):
        self.df=pd.read_csv(csv_file)
        self.data_root=data_root
        self.inputsize=inputsize
        
    def __getitem__(self, index):
        row=self.df.iloc[index]
        image_path=os.path.join(self.data_root,row['filename'])
        image=torch.load(image_path)
        image=cv2.resize(image,self.inputsize)
        label=row['label']
        return image,label

    def __len__(self):
        return len(self.df)
if __name__=="__main__":
    dataset=Mini_ImageNet_Datasets(csv_file=r"..\data\Mini-ImageNet-Dataset\train.csv",data_root=r"..\data\Mini-ImageNet-Dataset\train")
    print(len(dataset))
    image,label=dataset[0]
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroywindow("image")