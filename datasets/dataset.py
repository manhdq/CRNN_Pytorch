import random
import lmdb
import six
import sys
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, sampler


class textDataset(Dataset):
    def __init__(self, data_dir=None, data_csv=None, transform=None, target_transform=None):
        self.data_dir = data_dir

        data_df = pd.read_csv(data_csv)
        self.imageList = list(data_df['image name'])
        self.labelList = list(data_df['label'])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_dir = os.path.join(self.data_dir, self.imageList[index])
        try:
            img = Image.open(img_dir).convert('L')
        except:
            print('Corrupted image %s' % self.imageList[index])
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        label = self.labelList[index]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # Assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels