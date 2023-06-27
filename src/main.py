import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import utils as ut
from train import *
from dataset import load_train_data, load_test_data
import constants

def main(config):
    # Fixed random number seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Initialize image evaluation metrics
    best_psnr = 0.0
    best_ssim = 0.0

    if config.train.checkpoint.is_log:
        ut.log_on_train_start(log_name=config.exp_name, config=config)

    checkpoint_dir = os.path.join(constants.ROOT, 'model', config.exp_name) 
    ut.create_dir(checkpoint_dir)

    # Define basic elements for training
    netG, netD = define_model(config)

    # optimG = define_optimizer(netG, config)
    # optimD = define_optimizer(netD, config)
    
    optimG = optim.Adam(netG.parameters(), 
                       lr=config.train.optim.lr, 
                       betas=config.train.optim.betas)
    optimD = optim.Adam(netD.parameters(), 
                       lr=config.train.optim.lr, 
                       betas=config.train.optim.betas)


    schedulerG = define_scheduler(optimG, config)
    schedulerD = define_scheduler(optimD, config)

    if config.train.checkpoint.load_model:
        G_state_dict, optimG_state_dict, start_epoch = ut.load_checkpoint(config.train.checkpoint.gen)
        D_state_dict, optimD_state_dict, start_epoch = ut.load_checkpoint(config.train.checkpoint.disc)
        netG.load_state_dict(G_state_dict)
        netD.load_state_dict(D_state_dict)
        optimG.load_state_dict(optimG_state_dict)
        optimD.load_state_dict(optimD_state_dict)

    # Loss function
    content_criteria = nn.MSELoss()
    adversarial_criteria = nn.BCEWithLogitsLoss()
    feature_extractor = VGGLoss()
    feature_extractor = feature_extractor.to(constants.DEVICE)
    feature_extractor.eval()
    
    # Data loader
    print("Loading data ...")
    train_loader = load_train_data(root=config.train.dataset.data_dir, batch_size=config.train.hyp.batch_size)
    test_loader = load_test_data(hr_root=config.test.dataset.hr_dir, lr_root=config.test.dataset.lr_dir)
    print("Finish loading data")

    for epoch in range(config.train.hyp.num_epoch):
        netG.train()
        netD.train()
        D_loss, G_loss = train(
            train_loader, 
            epoch, 
            netG, 
            netD, 
            optimG, 
            optimD, 
            content_criteria, 
            adversarial_criteria, 
            feature_extractor, 
            config)

        schedulerD.step()
        schedulerG.step()

        psnr, ssim = test(test_loader, netG)
        is_best = psnr > best_psnr and ssim > best_psnr
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)

        print("D_loss: %.6f, G_loss: %.6f, psnr: %.6f, ssim: %.6f" % (D_loss, G_loss, psnr, ssim))

        ut.save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": netD.state_dict(),
                "optimizer": optimD.state_dict(),
            }, 
            f'{checkpoint_dir}/disc_{epoch+1}.pth.tar',
            f'{checkpoint_dir}/disc_best.pth.tar',
            is_best)

        ut.save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": netG.state_dict(),
                "optimizer": optimG.state_dict(),
            }, 
            f'{checkpoint_dir}/gen_{epoch+1}.pth.tar',
            f'{checkpoint_dir}/gen_best.pth.tar',
            is_best)

if __name__ == '__main__':
    main_config = ut.read_config(os.path.join(constants.ROOT,'config/config.yaml'))
    main(main_config)
