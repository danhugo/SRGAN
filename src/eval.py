import os
import torch
from tqdm import tqdm

from dataset import EvalDataset
import constants
from srgan import Generator
import utils as ut
from metrics import Metrics

from torchvision import transforms

def eval(config):
    netG = Generator().to(constants.DEVICE)
    G_state_dict, _, _ = ut.load_checkpoint(config.eval.checkpoint.load_dir)
    netG.load_state_dict(G_state_dict)
    netG.eval()
    metrics = Metrics(extract_y_channel=True)

    for set in constants.EVAL_SETS:
        mean_psnr = []
        mean_ssim = []
        names = []
        rescales = []
        eval_dataset = EvalDataset(root_dir=os.path.join(constants.ROOT, config.eval.dataset.data_dir, set))
        with torch.no_grad():
            for index, _ in enumerate(tqdm(eval_dataset.data, desc=f'Evaluation dataset {set}')):
                highres, lowres, name = eval_dataset[index] 
                highres = highres.to(constants.DEVICE)
                lowres = lowres.to(constants.DEVICE)

                rescale = netG(lowres)

                
                highres = highres * torch.tensor(constants.IMAGENET_STD)[:,None,None].to(constants.DEVICE) + torch.tensor(constants.IMAGENET_MEAN)[:,None,None].to(constants.DEVICE)
                rescale = rescale * torch.tensor(constants.IMAGENET_STD)[:,None,None].to(constants.DEVICE) + torch.tensor(constants.IMAGENET_MEAN)[:,None,None].to(constants.DEVICE)
                
                psnr = metrics.psnr(highres, rescale)
                ssim = metrics.ssim(highres, rescale)
                mean_psnr.append(psnr)
                mean_ssim.append(ssim)
                print(f"{name},  psnr: {psnr}, ssim: {ssim}")
                
                rescales.append(rescale.cpu())
                names.append(name)

        mean_psnr = torch.mean(torch.FloatTensor(mean_psnr))
        mean_ssim = torch.mean(torch.FloatTensor(mean_ssim))

        print(f'{set}: psnr: {mean_psnr} ssim: {mean_ssim}')
        ut.save_list_image(rescales, filenames=names, dir=os.path.join(constants.ROOT, 'data/eval', set))

if __name__ == "__main__":
    config = ut.read_config(os.path.join(constants.ROOT,'config/config.yaml'))
    eval(config)