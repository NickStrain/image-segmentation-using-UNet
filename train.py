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
