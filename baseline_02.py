# %%
import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from glob import glob
import gc
import time
from collections import defaultdict
import  matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy
import cv2

#pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.cuda import amp
import torch.optim as optim
import albumentations as A
import segmentation_models_pytorch as smp

from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import tifffile as tiff

# %% 

class Config:
    def __init__(self):
        self.seed = 42
        self.debug = False
        self.exp_name = 'baseline'
        self.comment = 'unet-efficient_b1-512x512'
        self.output_dir = './'
        self.model_name = 'Unet'
        self.backbone = ['efficientnet-b1','se_resnext50_32x4d']
        self.train_bs = 16
        self.valid_bs = 32
        self.img_size = [768,512]
        self.epochs = 30
        self.n_accumulate = max(1, 64 // self.train_bs)
        self.lr = 2e-3
#         self.lr = 6e-5
        self.scheduler = 'CosineAnnealingLR'
        self.min_lr = 1e-6
        self.T_max = int(2279 / (self.train_bs * self.n_accumulate) * self.epochs) + 50
        self.T_0 = 25
        self.warmup_epochs = 0
        self.wd = 1e-6
        self.n_fold = 5
        self.num_classes = 1
        self.input_channels = 3
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.gt_df = "/kaggle/input/sennet-hoa-gt-data/gt.csv"
        self.data_root = "/kaggle/input/blood-vessel-segmentation"
        self.train_groups = ["kidney_1_dense"]
        self.valid_groups = ["kidney_3_dense"]
        self.loss_func = "DiceLoss"
        
        
        self.data_transforms = {
            "train": A.Compose([
                A.Resize(*self.img_size, interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5),
                A.RandomScale(scale_limit=(0.0, 1.25), interpolation=cv2.INTER_CUBIC, p=0.5),
                A.RandomCrop(*self.img_size, p=1.0),], p=1.0),
            "valid": A.Compose([
                A.Resize(*self.img_size, interpolation=cv2.INTER_NEAREST), ], p=1.0)   
        }
        
        self.optimizers = 'adam'
config = Config()

def set_seed(seed):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(config.seed)

# %%

train_images_path = os.path.join(config.data_root, 'train', config.train_groups[0], 'images')
train_labels_path = os.path.join(config.data_root,'train', config.train_groups[0], 'labels')

print(train_images_path)

# %%
image_files = sorted([os.path.join(train_images_path, f) for f in os.listdir(train_images_path) if f.endswith('.tif')])
label_files = sorted([os.path.join(train_labels_path, f) for f in os.listdir(train_labels_path) if f.endswith('.tif')])

def show_images(images,titles= None, cmap='gray'):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(20, 10))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx], cmap=cmap)
        if titles:
            ax.set_title(titles[idx])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

first_image = tiff.imread(image_files[981])
first_label = tiff.imread(label_files[981])

show_images([first_image, first_label], titles=['Train Image', 'Train Label'])

# %%

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[...,None], [1, 1, 3]) #  Converts a grayscale image to an 
    #RGB image by replicating the single-channel image three times along the third axis.
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # Normalizes the image by dividing each pixel value by the maximum value, scaling it to the range [0, 1].
    return img

def load_msk(path):
    msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    msk = msk.astype('float32')
    msk/=255.0
    return msk



class DatasetBuilder(Dataset):
    def __init__(self,images,masks,input_size=(256,256),transforms=None):
        self.images = images
        self.masks = masks
        self.input_size = input_size
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = load_img(image)
        
   
        mask = self.masks[idx]
        mask= load_msk(mask)   
        
        if self.transforms:
            data = self.transforms(image=image, mask=mask)
            image  = data['image']
            mask  = data['mask']
            image = np.transpose(image, (2, 0, 1))  #Transposes the image array to have the channel dimension as the first dimension. This is a common format for PyTorch.
        return torch.tensor(image), torch.tensor(mask)


# %%

