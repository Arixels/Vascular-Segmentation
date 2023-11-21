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
import cv2 as cv
from PIL import Image
import numpy as np
img_path = "C:/Users/chang/source/repos/Vascular-Segmentation/data/train/kidney_1_dense/images/0000.tif"
try:
    sample_image = cv2.imread("/kaggle/input/test/kidney_5/images/0000.tif")
except FileNotFoundError:
    sample_image = cv2.imread("C:/Users/chang/source/repos/Vascular-Segmentation/data/train/kidney_1_dense/images/0000.tif")

pil_image = np.array(Image.open(img_path))

print(sample_image)
print(pil_image)

# %%
import matplotlib.pyplot as plt
def display(im , augments = False):
    
    img = im
    
    if augments :
        
        img = A.Compose([
        A.Resize(width = 512 , height = 512) , 
        A.Normalize(
            mean = [0 , 0 , 0] , 
            std = [1 , 1 , 1] , 
            max_pixel_value = 255
        ) , 
        ToTensorV2()
    ])(image = im)["image"]

    # return image
    
    plt.imshow(tf.reshape(img , (512 , 512 , 3)))
    
print("Test Image before preprocessing : ")
display(pil_image)
# %%
