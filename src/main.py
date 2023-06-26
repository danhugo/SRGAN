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
    # netG, netD = define_model(config)
    # optimG = define_optimizer(netG, config)
    # optimD = define_optimizer(netD, config)

    netG = srgan.Generator(config.model.nf, config.model.num_res_blocks).to(constants.DEVICE)
    netD = srgan.Discriminator(config.model.nf).to(constants.DEVICE)
    optimG = optim.Adam(netG.parameters(), 
                       lr=config.train.optim.lr, 
                       betas=config.train.optim.betas)
    optimD = optim.Adam(netG.parameters(), 
                       lr=config.train.optim.lr, 
                       betas=config.train.optim.betas)


    # schedulerG = define_scheduler(optimG, config)
    # schedulerD = define_scheduler(optimD, config)

    # if config.train.checkpoint.load_model:
    #     netG, optimG, schedulerG = ut.load_checkpoint(config.train.checkpoint.gen, netG, optimG, schedulerG)
    #     netD, optimD, schedulerD = ut.load_checkpoint(config.train.checkpoint.disc, netD, optimD, schedulerD)

    # Loss function
    pixel_criteria = nn.MSELoss()
    adversarial_criteria = nn.BCEWithLogitsLoss()
    feature_criteria = VGGLoss()
    feature_criteria = feature_criteria.to(constants.DEVICE)
    feature_criteria.eval()
    
    # Data loader
    print("Loading data ...")
    train_loader = load_train_data(root=config.train.dataset.data_dir, batch_size=config.train.hyp.batch_size)
    test_loader = load_test_data(hr_root=config.test.dataset.hr_dir, lr_root=config.test.dataset.lr_dir)
    print("Finish loading data")

    for epoch in range(config.train.hyp.num_epoch):
        # D_loss, G_loss = train(train_loader, epoch, netG, netD, optimG, optimD,pixel_criteria, adversarial_criteria, feature_criteria, config)
        netG.train()
        netD.train()

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
            adversarial_loss = 1e-3 * adversarial_criteria(output, torch.ones_like(output))
            # pixel_loss = pixel_criteria(fake, highres)
            hr_feature = feature_criteria(highres)
            sr_feature = feature_criteria(fake)

            content_loss = 0.006 * pixel_criteria(sr_feature, hr_feature.detach())
            # content_loss = 0.006 * feature_criteria(fake, highres.detach()) # 0.006
            G_loss = content_loss + adversarial_loss
            print(f"content {content_loss}, adv {adversarial_loss}")

            
            G_loss.backward()
            optimG.step()

        #     # Train Discriminator
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
        # schedulerD.step()
        # schedulerG.step()

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