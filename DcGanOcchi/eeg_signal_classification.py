# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")
# Dataset options

#Data - Data needs to be pre-filtered and filtered data is available

### BLOCK DESIGN ###
#Data
parser.add_argument('-edocchi', '--eeg_dataset_occhi', default="/home/d.sorge/eeg_visual_classification/dcgan/dataset_eeg_occhi/X_128el_16overlap.npy", help="EEG dataset occhi path")
parser.add_argument('-ld', '--label_dataset_occhi', default="/home/d.sorge/eeg_visual_classification/dcgan/dataset_eeg_occhi/Y_128el_16overlap.npy", help="EEG occhi labels path")

#Splits
parser.add_argument('-sp', '--splits-path', default="/projects/data/classification/eeg_cvpr_2017/block_splits_by_image_all.pth", help="splits path") #All subjects
#parser.add_argument('-sp', '--splits-path', default=r"data\block\block_splits_by_image_single.pth", help="splits path") #Single subject
### BLOCK DESIGN ###

parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number") #leave this always to zero.

#Subject selecting
parser.add_argument('-sub','--subject', default=0   , type=int, help="choose a subject from 1 to 6, default is 0 (all subjects)")

#Time options: select from 20 to 460 samples from EEG data
parser.add_argument('-tl', '--time_low', default=40, type=float, help="lowest time value")
parser.add_argument('-th', '--time_high', default=480,  type=float, help="highest time value")

# Model type/options
parser.add_argument('-mt','--model_type', default='lstm', help='specify which generator should be used: lstm|EEGChannelNet')
# It is possible to test out multiple deep classifiers:
# - lstm is the model described in the paper "Deep Learning Human Mind for Automated Visual Classification”, in CVPR 2017
# - model10 is the model described in the paper "Decoding brain representations by multimodal learning of neural activity and visual features", TPAMI 2020
parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
parser.add_argument('--pretrained_net', default='', help="path to pre-trained net (to continue training)")
parser.add_argument('--log_dir', default='', help="(to continue training)")
parser.add_argument('--name', default='', type=str)

# Training options
parser.add_argument("-b", "--batch_size", default=16, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=200, type=int, help="training epochs")
parser.add_argument('-osz', '--output_size', default=128, type=int, help="lstm output size")

# Save options
parser.add_argument('-sc', '--saveCheck', default=1, type=int, help="learning rate")

# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")

# Parse arguments
opt = parser.parse_args()
print(opt)

# Imports
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from scipy import signal
import numpy as np
import models
import importlib
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime

# Imports for Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

# Imports for eeg occhi
from sklearn.model_selection import train_test_split
from fast_ml.model_development import train_valid_test_split
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, eeg_path, label_path, split_name = "train"):
        # Load EEG signals
        eeg_data = np.load(eeg_path)
        # Load EEG labels
        label_data = np.load(label_path)

        # set aside 20% of train and test data for evaluation
        x_train, x_test, y_train, y_test = train_test_split(eeg_data, label_data,
            test_size=0.2, shuffle = True, random_state = 8)

        # Use the same function above for the validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
            test_size=0.25, random_state= 8) # 0.25 x 0.8 = 0.2        
        
        # modes - train, val and test
        if split_name == 'train':
            self.x_data, self.y_data = x_train, y_train
        elif split_name == 'val':
            self.x_data, self.y_data = x_val, y_val
        else:
            self.x_data, self.y_data = x_test, y_test

        # Compute size
        self.size = len(self.x_data)

    # Get size
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        return self.x_data[i].astype(np.float32), self.y_data[i]

# Create loader
loaders = {split: DataLoader(MyDataset(eeg_path=opt.eeg_dataset_occhi, label_path=opt.label_dataset_occhi, split_name = split), batch_size = opt.batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}

# Load model
model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for (key, value) in [x.split("=") for x in opt.model_params]}

# Create discriminator model/optimizer
module = importlib.import_module("models." + opt.model_type)
model = module.Model(**model_options)
optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr = opt.learning_rate)

      
# Setup CUDA
if not opt.no_cuda:
    model.cuda()
    print("Copied to CUDA")

if opt.pretrained_net != '':
        model = torch.load(opt.pretrained_net)
        print(model)
        
        log_dir = opt.log_dir

        train_logdir = os.path.join(log_dir, 'train')
        train_writer = SummaryWriter(train_logdir)

        val_logdir = os.path.join(log_dir, 'val')
        val_writer = SummaryWriter(val_logdir)

        test_logdir = os.path.join(log_dir, 'test')
        test_writer = SummaryWriter(test_logdir)
else:
    # create new log dir
    log_dir = os.path.join("logs/fit", opt.name, datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_logdir = os.path.join(log_dir, 'train')
    os.makedirs(train_logdir, exist_ok=True)
    train_writer = SummaryWriter(train_logdir)

    val_logdir = os.path.join(log_dir, 'val')
    os.makedirs(val_logdir, exist_ok=True)
    val_writer = SummaryWriter(val_logdir)

    test_logdir = os.path.join(log_dir, 'test')
    os.makedirs(test_logdir, exist_ok=True)
    test_writer = SummaryWriter(test_logdir)

#initialize training,validation, test losses and accuracy list
losses_per_epoch={"train":[], "val":[],"test":[]}
accuracies_per_epoch={"train":[],"val":[],"test":[]}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
# Start training

predicted_labels = [] 
correct_labels = []

for epoch in range(1, opt.epochs+1):
    # Initialize loss/accuracy variables
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    # Adjust learning rate for SGD
    if opt.optim == "SGD":
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Process each split
    for split in ("train", "val", "test"):
        total_steps = len(loaders[split])
        # Set network mode
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)  
        # Process all split batches
        for step, (input, target) in enumerate(tqdm(loaders[split])):
            # Check CUDA
            if not opt.no_cuda:
                input = input.to("cuda") 
                target = target.to("cuda") 
            # Forward
            print("input shape", input.shape)
            print("target shape", target.shape)
            output = model(input)
            print("out shape", output.shape)

            # Compute loss
            loss = F.cross_entropy(output, target)
            losses[split] += loss.item()
            # Compute accuracy
            _,pred = output.data.max(1)
            correct = pred.eq(target.data).sum().item()
            accuracy = correct/input.data.size(0)   
            accuracies[split] += accuracy
            counts[split] += 1
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        if split == "train":
            train_writer.add_scalar("Loss/epoch", losses[split] / total_steps, epoch + 1)
            train_writer.add_scalar("Accuracy/epoch", accuracies[split]/counts[split], epoch + 1)
        elif split == "val":
            val_writer.add_scalar("Loss/epoch", losses[split] / total_steps, epoch + 1)
            val_writer.add_scalar("Accuracy/epoch", accuracies[split]/counts[split], epoch + 1)
        else:
            test_writer.add_scalar("Loss/epoch", losses[split] / total_steps, epoch + 1)
            test_writer.add_scalar("Accuracy/epoch", accuracies[split]/counts[split], epoch + 1)
    
    # Print info at the end of the epoch
    if accuracies["val"]/counts["val"] >= best_accuracy_val:
        best_accuracy_val = accuracies["val"]/counts["val"]
        best_accuracy = accuracies["test"]/counts["test"]
        best_epoch = epoch
    
    TrL,TrA,VL,VA,TeL,TeA = losses["train"]/counts["train"],accuracies["train"]/counts["train"],losses["val"]/counts["val"],accuracies["val"]/counts["val"],losses["test"]/counts["test"],accuracies["test"]/counts["test"]
    print("Model: {11} - Subject {12} - Time interval: [{9}-{10}]  [{9}-{10} Hz] - Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}, TeA at max VA = {7:.4f} at epoch {8:d}".format(epoch,
                                                                                                         losses["train"]/counts["train"],
                                                                                                         accuracies["train"]/counts["train"],
                                                                                                         losses["val"]/counts["val"],
                                                                                                         accuracies["val"]/counts["val"],
                                                                                                         losses["test"]/counts["test"],
                                                                                                         accuracies["test"]/counts["test"],
                                                                                                         best_accuracy, best_epoch, opt.time_low, opt.time_high, opt.model_type, opt.subject))
    losses_per_epoch['train'].append(TrL)
    losses_per_epoch['val'].append(VL)
    losses_per_epoch['test'].append(TeL)
    accuracies_per_epoch['train'].append(TrA)
    accuracies_per_epoch['val'].append(VA)
    accuracies_per_epoch['test'].append(TeA)

    if epoch%opt.saveCheck == 0:
                state_dict = model.state_dict()
                torch.save(state_dict, '%s_%d_subject%d_epoch_%d.pth' % (opt.model_type, opt.output_size, opt.subject, epoch))

train_writer.close()
val_writer.close()
test_writer.close()
