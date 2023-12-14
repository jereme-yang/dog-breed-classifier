# dataset : https://www.kaggle.com/c/dog-breed-identification/data

import numpy as np
import pandas as pd
import cv2
import os
from constants import *

def load_all_images():
    df = pd.read_csv('.\data\labels.csv')
    x_train_dir = "./data/train"

    all_ims = []
    all_labs = []

    for _, (image_name, breed) in enumerate(df[['id', 'breed']].values):
        im_dir = os.path.join(x_train_dir, image_name + ".jpg")
        print(im_dir)

        im = cv2.imread(im_dir)
        resized = cv2.resize(im,IMAGE_SIZE, interpolation= cv2.INTER_AREA)
        all_ims.append(resized)
        all_labs.append(breed)

    np.save("./tmp/all_images.npy", all_ims)
    np.save("./tmp/all_labels.npy", all_labs)

def load_cat_images():
    df = pd.read_csv('.\data\labels.csv')
    x_train_dir = "./data/train"

    all_ims = []
    all_labs = []
    labels = set()
    max = 120
    for _, (image_name, breed) in enumerate(df[['id', 'breed']].values):
        if breed in labels:
            continue
        im_dir = os.path.join(x_train_dir, image_name + ".jpg")
        im = cv2.imread(im_dir)
        resized = cv2.resize(im,IMAGE_SIZE, interpolation= cv2.INTER_AREA)
        
        all_ims.append((resized))
        all_labs.append(breed)
        labels.add(breed)
        if len(labels) >= 120:
            break

    np.save("./tmp/cat_images.npy", all_ims)
    np.save("./tmp/cat_labels.npy", all_labs)
    print(all_labs)