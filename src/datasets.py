import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
class Mini_ImageNet_Datasets(Dataset):
    def __init__(self,csv_file,data_root,inputsize=(224,224),label_to_index_map=None):
        self.df=pd.read_csv(csv_file)
        self.data_root=data_root
        self.inputsize=inputsize
        if label_to_index_map==None:
            label_classes=self.df['label'].unique()
            self.label_to_index_map={label:i for i,label in enumerate(label_classes)}
        else:
             self.label_to_index_map=label_to_index_map
    def get_map(self):
        return self.label_to_index_map
    def __getitem__(self, index):
        row=self.df.iloc[index]
        image_path=os.path.join(self.data_root,row["label"],row['filename'])
        image=plt.imread(image_path)
        norm_mean = (0.5, 0.5, 0.5)
        norm_std = (0.5, 0.5, 0.5)
        image=A.Resize(height=self.inputsize[0],width=self.inputsize[1])(image=image)['image']
        image=A.Normalize(mean=norm_mean,std=norm_std)(image=image)['image']
        image=ToTensorV2()(image=image)['image']
        
        label=row['label']
  
        label=self.label_to_index_map[label]
        return image,label

    def __len__(self):
        return len(self.df)
if __name__=="__main__":
    dataset=Mini_ImageNet_Datasets(csv_file=r"..\data\Mini-ImageNet-Dataset\train.csv",data_root=r"..\data\Mini-ImageNet-Dataset\train")
    print(len(dataset))
    image,label=dataset[0]
    image=image.permute(1,2,0).numpy()
    image=(image*0.5+0.5)*255
    image=image.astype(np.uint8)
    plt.imshow(image)
    plt.show()

