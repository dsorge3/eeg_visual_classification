from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import cfg
from eegDatasetClass import EEGDataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import lstm
from utils.utils import make_grid, save_image
from pytorch_gan_metrics.utils import get_inception_score_from_directory
import random

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

args = cfg.parse_args()

def save_samples(train_loader, epoch, gen_net: nn.Module, lstm: nn.Module, clean_dir=True):
    # eval mode
    gen_net.eval()
    with torch.no_grad():
        os.makedirs(f"./training_output_128lstm2class/outputEpoch{epoch}", exist_ok=True)
        for i, (eeg, label, imgs) in enumerate(train_loader):
            eeg = eeg.to(torch.device("cuda:0"))
            rec = lstm(eeg, return_eeg_repr=True)
            #z = torch.normal(mean=0, std=1, size=rec.shape).cuda()      
            #rec_z = torch.cat((rec,z), dim=-1)
            rec = rec.view(-1, args.nz, 1, 1)                          
            sample_img = gen_net(rec)                                 
            save_image(sample_img, f'./training_output_128lstm2class/outputEpoch{epoch}/sampled_image_{i}_{epoch}.png', nrow=10, normalize=True, scale_each=True)
    return 0

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.ngf * 2, args.ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.ngf * 2, args.ngf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.ngf, args.ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.ngf, args.ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            
            # state size. (nc) x 32 x 32
            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
        )
        self.main2 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 8, args.ndf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf * 2, args.ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf * 2, args.ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main3 = nn.Sequential(
            nn.Conv2d(args.ndf * 8 + int(args.nz/2), args.ndf * 8, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(args.ndf * 8, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, input, rec):
        x = self.main1(input)
        x = self.main2(x)

        rec = rec.repeat((1, 1, x.shape[-2], x.shape[-1]))
        #print("rec", rec.shape)
        #print("x", x.shape)
        x = torch.cat([x, rec], axis=1)
        return self.main3(x)

def eval(netG, netD, lstm_net, criterion, device, real_label, fake_label, writer, eval_loader, eval_data, type, epoch):
    ds_test = tqdm.tqdm(eval_loader)
    G_losses = 0.0
    D_losses = 0.0
    total_steps = len(eval_loader)

    netG.eval()
    netD.eval()
    with torch.no_grad():
        total = 0.0
        correct = 0.0
        for step, (x, y, img) in enumerate(ds_test):

            eeg = x.to(device)  # SOSTITUITO CON L'EEG dal dataset
            with torch.no_grad():                             
                rec = lstm_net(eeg, return_eeg_repr=True)
            rec = rec.view(-1, int(args.nz/2), 1, 1)

            z = torch.normal(mean=0, std=1, size=rec.shape).cuda()   
            rec_z = torch.cat((rec, z), dim=1)

            # Format batch
            real_cpu = img.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu, rec).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            D_x = output.mean().item()

            ## Train with all-fake batch
           
            # Generate fake image batch with G
            fake = netG(rec_z)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach(), rec).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            
            imgs = torch.empty_like(real_cpu)
            for i in range(eeg.shape[0]):
                _, _, wimg = eval_data[random.randint(0, len(eval_data) - 1)]
                imgs[i] = wimg

            output = netD(imgs.to(device), rec).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_real_wrong = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            
            D_G_z_real_wrong = output.mean().item()

            errD = errD_real + errD_fake + errD_real_wrong
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, rec).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            D_G_z2 = output.mean().item()

            # Save Losses for plotting later
            G_losses += errG.cpu().item()
            D_losses += errD.cpu().item()

            ds_test.set_description(f'[{type}] Loss_G: {G_losses / (step + 1):.4f} - Loss_D: {D_losses / (step + 1):.4f}')

    writer.add_scalar("Loss_G/epoch", G_losses / total_steps, epoch + 1)
    writer.add_scalar("Loss_D/epoch", D_losses / total_steps, epoch + 1)

def main():

    Dt = EEGDataset
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    

    train_data = Dt(eeg_signals_path=args.eeg_dataset, split_path=args.splits_path, split_num=args.split_num, transform=transform)
    val_data = Dt(eeg_signals_path=args.eeg_dataset, split_path=args.splits_path, split_num=args.split_num, transform=transform, split_name="val")
    test_data = Dt(eeg_signals_path=args.eeg_dataset, split_path=args.splits_path, split_num=args.split_num, transform=transform, split_name="test")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator().to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)
    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator().to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    # Print the model
    print(netD)

    # Lstm Net
    lstm_net = lstm.Model(128, 128, 1, 128)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    start_epoch = 0

    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)

        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch'] + 1

        netG.load_state_dict(checkpoint['modelG'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])

        netD.load_state_dict(checkpoint['modelD'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])

        log_dir = args.log_dir
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

        print(f'=> resuming lstm from {args.lstm_path}')      # RESUME LSTM NET
        assert os.path.exists(args.lstm_path)
        checkpoint_lstm = os.path.join(args.lstm_path)
        assert os.path.exists(checkpoint_lstm)
        loc = 'cuda:{}'.format(args.gpu)
        #lstm_dict = torch.load(checkpoint_lstm, map_location=loc)
        lstm_dict = torch.load(checkpoint_lstm, map_location='cpu')
        lstm_net.load_state_dict(lstm_dict)
        lstm_net.zero_grad()
        lstm_net.eval()
        #lstm_net.to(torch.device("cuda"))
        lstm_net.to(device)    
        #lstm_net = torch.nn.parallel.DistributedDataParallel(lstm_net, device_ids=[args.gpu], find_unused_parameters=False)
        print(f'=> loaded checkpoint {checkpoint_lstm}')

        del checkpoint
    else:
        # create new log dir
        log_dir = os.path.join("logs/fit", args.name, datetime.now().strftime("%Y%m%d-%H%M%S"))

        print(f'=> resuming lstm from {args.lstm_path}')      # RESUME LSTM NET
        assert os.path.exists(args.lstm_path)
        checkpoint_lstm = os.path.join(args.lstm_path)
        assert os.path.exists(checkpoint_lstm)
        loc = 'cuda:{}'.format(args.gpu)
        #lstm_dict = torch.load(checkpoint_lstm, map_location=loc)
        lstm_dict = torch.load(checkpoint_lstm, map_location='cpu')
        lstm_net.load_state_dict(lstm_dict)
        lstm_net.zero_grad()
        lstm_net.eval()
        #lstm_net.to(torch.device("cuda")) 
        lstm_net.to(device)  
        #lstm_net = torch.nn.parallel.DistributedDataParallel(lstm_net, device_ids=[args.gpu], find_unused_parameters=False)
        print(f'=> loaded checkpoint {checkpoint_lstm}')

    train_logdir = os.path.join(log_dir, 'train')
    os.makedirs(train_logdir, exist_ok=True)
    train_writer = SummaryWriter(train_logdir)

    val_logdir = os.path.join(log_dir, 'val')
    os.makedirs(val_logdir, exist_ok=True)
    val_writer = SummaryWriter(val_logdir)

    test_logdir = os.path.join(log_dir, 'test')
    os.makedirs(test_logdir, exist_ok=True)
    test_writer = SummaryWriter(test_logdir)

    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(int(start_epoch), int(args.num_epochs)):
        # For each batch in the dataloader
        ds_train = tqdm.tqdm(train_dataloader)
        G_losses = 0.0
        D_losses = 0.0
        total_steps = len(train_dataloader)
        total = 0.0
        for step, (x, y, img) in enumerate(ds_train):

            eeg = x.to(device)  # SOSTITUITO CON L'EEG dal dataset
            with torch.no_grad():                             
                rec = lstm_net(eeg, return_eeg_repr=True)
            rec = rec.view(-1, int(args.nz/2), 1, 1)

            z = torch.normal(mean=0, std=1, size=rec.shape).cuda()   
            rec_z = torch.cat((rec, z), dim=1)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = img.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu, rec).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch

            # Generate fake image batch with G
            fake = netG(rec_z)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach(), rec).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            
            # Calculate Discriminator over real_image and wrong eeg

            # random rec
            #wrong_label = torch.empty_like(y.cpu())
            #eeg = torch.empty_like(eeg.cpu())
            imgs = torch.empty_like(real_cpu)
            for i in range(eeg.shape[0]):
                _, _, wimg = train_data[random.randint(0, len(train_data) - 1)]
                imgs[i] = wimg

            output = netD(imgs.to(device), rec).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_real_wrong = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_real_wrong.backward()

            D_G_z_real_wrong = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake + errD_real_wrong
            
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, rec).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(wx): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, args.num_epochs, step, len(train_dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z_real_wrong, D_G_z2))

            # Save Losses for plotting later
            G_losses += errG.cpu().item()
            D_losses += errD.cpu().item()

            iters += 1
        
        train_writer.add_scalar("Loss_G/epoch", G_losses / total_steps, epoch + 1)
        train_writer.add_scalar("Loss_D/epoch", D_losses / total_steps, epoch + 1)

        eval(netG, netD, lstm_net, criterion, device, real_label, fake_label, val_writer, val_dataloader, val_data, type="Val", epoch=epoch)
        eval(netG, netD, lstm_net, criterion, device, real_label, fake_label, test_writer, test_dataloader, test_data, type="Test", epoch=epoch)

        save_samples(ds_train, epoch, netG, lstm_net)
        IS, IS_std = get_inception_score_from_directory(f'/home/d.sorge/eeg_visual_classification/dcgan/training_output_128lstm2class/outputEpoch{epoch}')
        print("Inception Score Epoch", epoch, ":", IS)

        save_dict = {
            'modelG': netG.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'modelD': netD.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'epoch': epoch,
        }

        torch.save(save_dict, os.path.join(log_dir, f"checkpoint_{epoch}.pth"))
    
    train_writer.close()
    val_writer.close()
    test_writer.close()


if __name__ == '__main__':
    main()
