# %%

import os
import numpy 
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp



# %%
try: 
    train_dir = "/kaggle/input/hubmap-hacking-the-human-vasculature/train"
    test_dir = "/kaggle/input/hubmap-hacking-the-human-vasculature/test"
except FileNotFoundError: 
    train_dir = "C:/Users/chang/source/repos/Vascular-Segmentation/data/train"
    test_dir = "C:/Users/chang/source/repos/Vascular-Segmentation/data/test"
    
        
# %%
try:
    with open("/kaggle/input/hubmap-hacking-the-human-vasculature/sample_submission.csv" , "r") as f:
        k = list(f)
    print(k[0])
except FileNotFoundError:
    with open("C:/Users/chang/source/repos/Vascular-Segmentation/data/sample_submission.csv" , "r") as f:
        k = list(f)
    print(k[0])


# %%
# import cv2 as cv
# from PIL import Image

# def preprocess(image):
    
#     image = np.clip(image , width , height)

#     image = (image - image.mean()) / image.std()

#     return image
import cv2

import tensorflow as tf

import albumentations as A
from albumentations.pytorch import ToTensorV2
A.Compose([
    A.Resize(width = 512, height = 512),
    A.Normalize(
        mean = [0, 0], 
        std = [1, 1], 
        max_pixel_value = 255
    ),
    ToTensorV2()
])

# %%
try:
    sample_image = cv2.imread("/kaggle/input/hubmap-hacking-the-human-vasculature/test/72e40acccadf.tif")
except FileNotFoundError:
    sample_image = cv2.imread("C:\Users\chang\source\repos\Vascular-Segmentation\data\test\kidney_5\images\0000.tif")
