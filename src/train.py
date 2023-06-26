import torch
import torch.optim as optim
import torch.nn as nn

from easydict import EasyDict as esdict
from tqdm import tqdm

import utils as ut
import constants
import srgan as srgan
from vgg_loss import VGGLoss
from metrics import Metrics

def define_model(config):
    netG = srgan.Generator(config.model.nf, config.model.num_res_blocks).to(constants.DEVICE)
    netD = srgan.Discriminator(config.model.nf).to(constants.DEVICE)
    netG.train()
    netD.train()
    return netG, netD

def define_optimizer(model, config):
    optimizer = optim.Adam(model.parameters(), 
                       lr=config.train.optim.lr, 
                       betas=config.train.optim.betas)
    return optimizer

def define_scheduler(model, config):
    scheduler = optim.lr_scheduler.MultiStepLR(model, 
                                               milestones=config.train.lr_scheduler.milestones, 
                                               gamma=config.train.lr_scheduler.gamma)
    return scheduler

def train(
        train_loader, 
        epoch: int,
        netG: nn.Module, 
        netD: nn.Module, 
        optimG: optim.Adam, 
        optimD: optim.Adam,
        content_criteria: nn.MSELoss,
        adversarial_criteria: nn.BCEWithLogitsLoss, 
        feature_extractor: VGGLoss,
        config: dict,
        ) -> None:

    mean_D_loss = []
    mean_G_loss = []

    log_in_epoch = esdict()
    num_batch = len(train_loader)

    for step, (highres, lowres) in enumerate(tqdm(train_loader, desc=f'{epoch+1}/{config.train.hyp.num_epoch}')):
        highres = highres.to(constants.DEVICE)
        lowres = lowres.to(constants.DEVICE)

        # Train Genenrator
        optimG.zero_grad()

        fake = netG(lowres)
        output = netD(fake)
        adversarial_loss = constants.ADVERSARIAL_LOSS_COEFF * adversarial_criteria(output, torch.ones_like(output))

        hr_feature = feature_extractor(highres)
        sr_feature = feature_extractor(fake)

        content_loss = constants.CONTENT_LOSS_COEFF * content_criteria(sr_feature, hr_feature.detach())

        pixel_loss = content_criteria(fake, highres)
        G_loss = content_loss + adversarial_loss + pixel_loss
        
        G_loss.backward()
        optimG.step()

        # Train Discriminator
        optimD.zero_grad()
        disc_real = netD(highres)
        D_real_loss = adversarial_criteria(disc_real, torch.ones_like(disc_real))
        
        disc_fake = netD(fake.detach())
        D_fake_loss = adversarial_criteria(disc_fake, torch.zeros_like(disc_fake))


        D_loss = D_fake_loss + D_real_loss
        D_loss.backward()
        optimD.step()

        # mean loss
        mean_D_loss.append(D_loss)
        mean_G_loss.append(G_loss)

        global_step = step + epoch * num_batch
        if global_step % config.train.log_iter == 0:
            log_in_epoch.D_loss = D_loss
            log_in_epoch.G_loss = G_loss
        
            if config.train.checkpoint.is_log:
                ut.log_image(ut.get_log_image(netG), global_step)
                ut.log_in_train_epoch(log_in_epoch, global_step)

    mean_D_loss = torch.mean(torch.FloatTensor(mean_D_loss)).cpu().item()
    mean_G_loss = torch.mean(torch.FloatTensor(mean_G_loss)).cpu().item()
    return mean_D_loss, mean_G_loss

def test(
        test_loader,
        netG: nn.Module,
        ):
    """Test in one epoch return PSNR, SSIM on test dataset.

    Returns: psnr (float), ssim (float)
    """
    
    netG.eval()
    metric = Metrics()
    psnres = []
    ssimes = []

    with torch.no_grad():
        for hr, lr in test_loader:
            hr = hr.to(constants.DEVICE)
            lr = lr.to(constants.DEVICE)
            rescale = netG(lr)
            psnr = metric.psnr(hr, rescale)
            ssim = metric.ssim(hr, rescale)

            psnres.append(psnr)
            ssimes.append(ssim)

    psnr = torch.mean(torch.FloatTensor(psnres)).cpu().item()
    ssim = torch.mean(torch.FloatTensor(ssimes)).cpu().item()
    netG.train()
            
    return psnr, ssim
