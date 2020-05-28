import glob
import random
import os
import sys
import numpy as np
import pandas as pd
import math
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import horisontal_flip


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, label_file, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        self.label_file = label_file
        self.data = pd.read_csv(label_file)
        # print(self.data.columns)
        # print(self.data.head(10))
        self.img_files = self.data['filename'].tolist()
        # delete duplicates from list
        self.img_files = list(dict.fromkeys(self.img_files))
        
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        filename = self.img_files[index % len(self.img_files)].rstrip()
        img_path = os.path.join(os.path.dirname(self.label_file), filename)

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_indexs = self.data.index[self.data['filename'] == filename].tolist()
        # print(self.data.index[self.data['filename'] == filename], filename)

        # shape: (idx, class, x, y, h, w)
        targets = torch.zeros((len(label_indexs), 6))
        # shape: (class, x, y, h, w)
        boxes = torch.zeros((len(label_indexs), 5))
        for i, index in enumerate(label_indexs):
            # Extract coordinates for unpadded + unscaled image
            x1 = self.data.iloc[index]['xmin']
            y1 = self.data.iloc[index]['ymin']
            x2 = self.data.iloc[index]['xmax']
            y2 = self.data.iloc[index]['ymax']
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[i, 1] = ((x1 + x2) / 2) / padded_w
            boxes[i, 2] = ((y1 + y2) / 2) / padded_h
            boxes[i, 3] = math.fabs(x2 - x1) * w_factor / padded_w
            boxes[i, 4] = math.fabs(y2 - y1) * h_factor / padded_h
            # print(boxes[i])

        targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    dataset = ListDataset(
        label_file='/home/haodong/Data/Gesture_Data/egohands/images/train/train_labels.csv',
        normalized_labels=False)
    print(dataset[0])
