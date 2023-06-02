import os
from  PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):
        self.image_path = image_dir
        self.mask_path = mask_dir
        self.transformer = transform
        self.image = os.listdir(image_dir)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir,self.image[index])
        mask_path = os.path.join(self.mask_path,self.image[index].replace('. jpg','_mask.gif'))
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        mask[mask == 255.0] = 1.0

        if self.transformer is not None:
            agumentation  = self.transformer(image=image,mask=mask)
            image = agumentation['image']
            mask = agumentation['mask']
        return image,mask
