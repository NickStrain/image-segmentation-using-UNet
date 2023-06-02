import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import unet

#Hyperparameter
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
num_epochs = 3
image_heigth = 160
image_weigth = 240
train_imgdir = 'E:\image segmentation\\train'
train_masdir = 'E:\image segmentation\\train_masks'
test_imgdir = "E:\image segmentation\\test"
test_masdir = 'E:\image segmentation\\test_mask'


def train_model(loader,model,optimizer,loss_fn,scaler):
    loop = tqdm(loader)
    for batch_idx,(data,targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)
        #forward
        pre = unet(data)
        loss= loss_fn(pre,targets)
        #backward
        optimizer.zero_grade()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
def main():
    train_transformer = A.Compose([
        A.Resize(height=image_heigth,width=image_weigth),
        A.Rotate(limit=35,p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0,0.0,0.0],
            std=[1.0,1.0,1.0],
            max_pixel_value=255.0),
        ToTensorV2(),
    ])

    test_transformer = A.Compose([
        A.Resize(height=image_heigth,width=image_weigth),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0),
        ToTensorV2(),
    ])

