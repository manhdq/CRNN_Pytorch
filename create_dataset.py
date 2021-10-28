# encoding: utf-8

import os
import cv2
import lmdb
import shutil
import random
import pandas as pd
import numpy as np

from utils import makedir


IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes


def create_from_txt(data_dir, data_txt, outputPath, valSplit):
    makedir(outputPath)

    imageList = []
    labelList = []

    with open(data_txt, 'r') as f:
        for line in f:
            line = line.rstrip().split(' ')
            if line[0].split('.')[-1].lower() not in IMG_FORMATS:
                continue
            imageList.append(line[0].split('/')[-1].strip())
            labelList.append(line[1].strip().lower())
    f.close()
    imageList = np.array(imageList)
    labelList = np.array(labelList)

    indices = list(range(len(imageList)))
    random.shuffle(indices)

    val_thres = int(np.floor(len(imageList) * valSplit))

    val_indices = indices[:val_thres]
    train_indices = indices[val_thres:]
    
    imageTrainList = imageList[train_indices]
    labelTrainList = labelList[train_indices]

    imageValList = imageList[val_indices]
    labelValList = labelList[val_indices]

    train_df = pd.DataFrame(list(zip(imageTrainList, labelTrainList)), columns=['image name', 'label'], index=None)

    val_df = pd.DataFrame(list(zip(imageValList, labelValList)), columns=['image name', 'label'], index=None)

    trainImageDir = os.path.join(outputPath, 'files/train')
    valImageDir = os.path.join(outputPath, 'files/test')

    makedir(trainImageDir)
    makedir(valImageDir)

    for img_f in imageTrainList:
        img_source = os.path.join(data_dir, img_f)
        shutil.copy(img_source, trainImageDir)

    for img_f in imageValList:
        img_source = os.path.join(data_dir, img_f)
        shutil.copy(img_source, valImageDir)

    train_df.to_csv(os.path.join(outputPath, 'train.csv'))
    val_df.to_csv(os.path.join(outputPath, 'test.csv'))


if __name__ == '__main__':
    create_from_txt(data_dir='datasets/data/OCR_raw_data/Synth25KFrom90K/images',
                    data_txt='datasets/data/OCR_raw_data/Synth25KFrom90K/annotations-training.txt',
                    outputPath='datasets/data/Synth25k',
                    valSplit=.2)