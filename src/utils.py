from PIL import Image
import os

import yaml
import wandb as wb
from easydict import EasyDict as esdict

import torch
from torch import nn
from torch.optim import Optimizer
from torchvision.utils import save_image, make_grid
from torchvision import transforms

import constants

# CONFIG
def read_config(path: str):
    """read configuration from `.yaml` file."""
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
        return esdict(config)

# CHECKPOINT
def save_checkpoint(
        state_dict: dict, 
        checkpoint_path: str, 
        best_ckpt_path: str,
        is_best=False,
        ):
    """Save model and optimizer state_dict to `.pth.tar file`."""
    torch.save(state_dict, checkpoint_path)

    if is_best:
        torch.save(state_dict, best_ckpt_path)

def load_checkpoint(
        checkpoint_path: str, 
        ):
    """Load model and optimizer state_dict from checkpoint file return (model_state_dict, optimizer_state_dict, epoch)"""
    checkpoint = torch.load(checkpoint_path, map_location=constants.DEVICE)
    model_state_dict = checkpoint["model"]
    optimizer_state_dict = checkpoint["optimizer"]
    epoch = checkpoint["epoch"]
    # scheduler = checkpoint["scheduler"]
    # psnr = checkpoint["psnr"]
    # ssim = checkpoint["ssim"]

    return model_state_dict, optimizer_state_dict, epoch

# LOG TRAINING
def log_on_train_start(log_name, config):
    """Initiate and start project if module is present"""
    wb.init(project=constants.NAME_PROJECT,
            name=log_name,
            config=config) if not wb.run else wb.run

def log_in_train_epoch(args: dict, step: int):
    """Log metrics each training epoch."""
    wb.run.log(args, step)

def log_end_train_epoch(args: dict, step: int):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(args, step)

def log_image(image: torch.Tensor, step: int, save=False, path= os.path.join(constants.ROOT, 'reports/rescale.jpg')):
    """Log image to wandb server and save (optional)."""
    pil_image = transforms.ToPILImage()(image)
    wb.run.log({'rescale' : wb.Image(pil_image)}, step)
    if save:
        save_image(image, path)

def get_log_image(gen):
    """Get log image for evaluation when training and save it to folder `reports`.
    - gen (nn.Module): The Generator model
    - return (tensor): size C x H x W grid image
    """
    files = os.listdir(os.path.join(constants.ROOT, 'reports/LR'))
    gen.eval()
    gen = gen.to(constants.DEVICE)

    log_image = []
    min_height = 1e4

    for file in files:
        img = Image.open(os.path.join(constants.ROOT, 'reports/LR', file))
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(constants.IMAGENET_MEAN, constants.IMAGENET_STD)
        ])
        with torch.no_grad():
            img = trans(img).unsqueeze(0).to(constants.DEVICE)
            upscaled_img = gen(img).squeeze()
            upscaled_img = upscaled_img * torch.tensor(constants.IMAGENET_STD)[:,None, None].to(constants.DEVICE) + torch.tensor(constants.IMAGENET_MEAN)[:, None, None].to(constants.DEVICE)
            upscaled_img = torch.clip(upscaled_img, min=0.0, max=1.0)

        log_image.append(upscaled_img)
        if upscaled_img.shape[-1] < min_height:
            min_height = upscaled_img.shape[-1]
        save_image(upscaled_img, os.path.join(constants.ROOT, 'reports/HR', f'rescale_{file}'))
    gen.train()
    resize = transforms.Resize((min_height, min_height))
    for i, img in enumerate(log_image):
        log_image[i] = resize(img)

    log_image = make_grid(torch.stack(log_image), nrow=len(files))
    return log_image
    
# DIR
def create_dir(dir):
    """Create dir if non existing"""
    if not os.path.exists(dir):
        os.mkdir(dir)

def save_list_image(images, filenames, dir):
    """Save list of Tensor Images or a Tensor of Batch Images.
    - images (list[Tensor] or Tensor): list of images
    - filenames (list): list of corresponding filenames. e.g 'cat'
    - dir (str): saved directory. 
    """
    for image, filename in list(zip(images, filenames)):
        save_image(image, f'{dir}/{filename}_SRGANX4.png')
            

