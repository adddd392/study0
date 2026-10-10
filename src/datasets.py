import os
import torch
from torch.utils.data import Dataset
import pandas as pd
class Mini_ImageNet_Datasets(Dataset):
    def __init__(self,csv_file,data_root,inputsize=(224,224)):
        self.df=pd.read_csv(csv_file)
        self.data_root=data_root
        self.inputsize=inputsize
        
    def __getitem__(self, index):
        row=self.df.iloc(index)
        image_path=os.join(self.data_root,row['filename'])

        
    def __len__(self):
        return len(self.df)