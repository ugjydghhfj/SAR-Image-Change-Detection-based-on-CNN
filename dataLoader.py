import os
import torch
import numpy as np
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self,img_path,lable_path):
        self.imageNameList=os.listdir(img_path)
        self.image=img_path
        self.lable=lable_path

    def __getitem__(self, item):
        image=np.load(self.image+self.imageNameList[item])
        lable=np.load(self.lable+self.imageNameList[item])

        image=torch.from_numpy(image).type(torch.FloatTensor)
        lable=torch.from_numpy(lable).type(torch.LongTensor)

        return image,lable

    def __len__(self):
        return len(self.imageNameList)