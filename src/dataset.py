
import sys
import os
sys.path.append(os.path.abspath('.'))

from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
import os
import random
from PIL import Image
from torchvision import transforms
from src import constants
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

class CustomDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            nsample: Optional[int] = None,
            transform: Optional[Callable] = None):
        """
        Obtain dataset to training SRGAN model. Return (high resolution image, low resolution image).
        
        Args:
        - root (str): image root directory.
        - nsample (int): number of sample retrived.
        - transform (callable): optional applied transform function for dataset.

        Attrs:
        - data (list[tuple]): list of image paths.
        """
        super(CustomDataset, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        
        categories = os.listdir(self.root_dir)

        for category in  categories:
            imgs = os.listdir(os.path.join(root_dir, category))
            self.data += list(os.path.join(root_dir, category, img) for img in imgs)
        
        if nsample is not None:
            random.seed(0)
            self.data = random.sample(self.data, nsample)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:
        """Return tuple tensor images contains (highres_img, lowres_img)."""
        img = Image.open(self.data[index]).convert('RGB')

        both_transform =transforms.Compose([
            transforms.RandomCrop((constants.HIGH_RES, constants.HIGH_RES)),
        ])

        highres_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=constants.IMAGENET_MEAN, std=constants.IMAGENET_STD),
        ])

        lowres_transform = transforms.Compose([
            transforms.Resize((constants.LOW_RES, constants.LOW_RES), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=constants.IMAGENET_MEAN, std=constants.IMAGENET_STD), 
        ])

        if self.transform is not None:
            img = self.transform(img)
        
        img = both_transform(img)
        
        highres_img, lowres_img = highres_transform(img), lowres_transform(img)

        return highres_img, lowres_img

class EvalDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            ):
        """
        Obtain dataset to training SRGAN model. Return (high resolution image, low resolution image). 
        
        Args:
        - root (str): image root directory.

        Attrs:
        - data (tuple (highres, lowres, name)): tuple of high resolution image (dir), low resolution image (dir) and name of image.
        """
        super(EvalDataset, self).__init__()
        self.data = []
        self.root_dir = root_dir

        imgs = os.listdir(os.path.join(root_dir))
        highres = sorted(list((os.path.join(root_dir, img)) for img in imgs if img.endswith('_HR.png')))
        lowres = sorted(list((os.path.join(root_dir, img)) for img in imgs if img.endswith('_LR.png')))
        name = sorted(list(img.split('_HR')[0] for img in imgs if img.endswith('_HR.png')))
        self.data += list(zip(highres, lowres, name))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:
        """Return tuple tensor images contains (highres_img, lowres_img, img_name)."""
        highres, lowres, name = Image.open(self.data[index][0]), Image.open(self.data[index][1]), self.data[index][2]
        transform =transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), 
        ])

        highres, lowres = transform(highres), transform(lowres)
        return highres, lowres, name

class TestDataset(Dataset):
    def __init__(
            self,
            hr_dir: str,
            lr_dir: str,
            ) -> None:
        """Return test dataset

        Args:
        - hr_dir (str): high resolution directory.
        - lr_dir (str): low resolution directory.
        
        Attrs:
        - data: list of tuple (high res path, low res path).

        """
        super().__init__()
        self.data = []

        hr = sorted(os.path.join(hr_dir, img) for img in os.listdir(hr_dir))
        lr = sorted(os.path.join(lr_dir, img) for img in os.listdir(lr_dir))
        self.data += list(zip(hr, lr))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:

        highres, lowres = Image.open(self.data[index][0]), Image.open(self.data[index][1])
        transform =transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=constants.IMAGENET_MEAN, std=constants.IMAGENET_STD), 
        ])

        highres, lowres = transform(highres), transform(lowres)
        return highres, lowres

def load_train_data(root: str, batch_size: int = 16):
    dataset = CustomDataset(root, constants.NUM_TRAIN_SAMPLE)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=constants.NUM_WORKERS)

def load_test_data(hr_root: str, lr_root: str):
    dataset = TestDataset(hr_root, lr_root)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=constants.NUM_WORKERS)
