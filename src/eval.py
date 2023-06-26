import os
import torch
from tqdm import tqdm

from dataset import EvalDataset
import constants
from srgan import Generator
import utils as ut
from metrics import Metrics

def eval(args):
    netG = Generator().to(constants.DEVICE)
    ut.load_checkpoint(args.eval_checkpoint_path, netG)
    netG.eval()
    metrics = Metrics(extract_y_channel=True)

    for set in constants.EVAL_SETS:
        mean_psnr = []
        mean_ssim = []
        names = []
        rescales = []
        eval_dataset = EvalDataset(root=os.path.join(constants.ROOT, 'data/eval', set))
        with torch.no_grad():
            for index, _ in enumerate(tqdm(eval_dataset.data, desc=f'Evaluation dataset {set}')):
                highres, lowres, name = eval_dataset[index] 
                highres = highres.to(constants.DEVICE)
                lowres = lowres.to(constants.DEVICE)

                rescale = netG(lowres)
                psnr = metrics.psnr(highres, rescale)
                ssim = metrics.ssim(highres, rescale)
                mean_psnr.append(psnr)
                mean_ssim.append(ssim)
                
                rescales.append(rescale)
                names.append(name)

        mean_psnr = torch.mean(torch.FloatTensor(mean_psnr))
        mean_ssim = torch.mean(torch.FloatTensor(mean_ssim))

        print(f'{set}: psnr: {mean_psnr} ssim: {mean_ssim}')
        ut.save_list_image(rescales, filenames=names, dir=os.path.join(constants.ROOT, 'eval', set))

if __name__ == "__main__":
    config = ut.read_config(os.path.join(constants.ROOT,'src/config.yaml'))
    eval(config)