import torch
import os

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
NAME_PROJECT = 'SRGAN'
ROOT = os.path.abspath('.')
HIGH_RES = 96
LOW_RES = HIGH_RES // 4
NUM_TRAIN_SAMPLE = 350000
PIXEL_VALUE_RANGE = 1.0 # eval image will be ranged in [0, 1]
EVAL_SETS = ['Set5', 'Set14', 'BSD100']
NUM_WORKERS = 1
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ADVERSARIAL_LOSS_COEFF = 1e-3
CONTENT_LOSS_COEFF = 0.006
