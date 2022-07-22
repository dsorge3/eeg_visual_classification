# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")
# Dataset options

#Data - Data needs to be pre-filtered and filtered data is available

### BLOCK DESIGN ###
#Data
parser.add_argument('-ed', '--eeg-dataset', default="/projects/data/classification/eeg_cvpr_2017/eeg_55_95_std.pth", help="EEG dataset path") #55-95Hz
#parser.add_argument('-ed', '--eeg-dataset', default="/projects/data/classification/eeg_cvpr_2017/eeg_5_95_std.pth", help="EEG dataset path") #5-95Hz
#parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_14_70_std.pth", help="EEG dataset path") #14-70Hz
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


# Dataset class
class EEGDataset:
    
    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if opt.subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        else:
            self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high,:]

        if opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1,128,opt.time_high-opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label

# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label

# Load dataset
dataset = EEGDataset(opt.eeg_dataset)    

# Create loaders
loaders = {split: DataLoader(Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = split), batch_size = opt.batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}

# Load model
model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for (key, value) in [x.split("=") for x in opt.model_params]}

# Create discriminator model/optimizer
module = importlib.import_module("models." + opt.model_type)
model = module.Model(**model_options)
optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr = opt.learning_rate)

if opt.pretrained_net != '':
        
        print(f'=> resuming lstm from {opt.pretrained_net}')      #MODIFICA: RESUME LSTM NET
        assert os.path.exists(opt.pretrained_net)
        checkpoint_lstm = os.path.join(opt.pretrained_net)
        assert os.path.exists(checkpoint_lstm)
        #loc = 'cuda:{}'.format(args.gpu)
        lstm_dict = torch.load(checkpoint_lstm, map_location='cpu')
        model.load_state_dict(lstm_dict)
        model.zero_grad()
        model.eval()
        #lstm_net = lstm_net.cuda()
        #lstm_net.to(torch.device("cuda"))
        #lstm_net = torch.nn.parallel.DistributedDataParallel(lstm_net, device_ids=[args.gpu], find_unused_parameters=False)
        print(f'=> loaded checkpoint {checkpoint_lstm}')

# **** COSTRUZIONE MATRICE CONFUSIONE ****
y_pred = []
y_true = []

for split in ("train", "val", "test"):
        # Set network mode
        if split == "test":
            model.eval()
            torch.set_grad_enabled(False)  
            # iterate over test data
            for inputs, labels in loaders[split]:
                    output = model(inputs) # Feed Network

                    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                    y_pred.extend(output) # Save Prediction
                        
                    labels = labels.data.cpu().numpy()
                    y_true.extend(labels) # Save Truth

# constant for classes
classes = ('Sorrel', 'Parachute', 'Iron', 'Anemone fish', 'Espresso maker',
        'Mug', 'Mountain bike', 'Revolver', 'Panda', 'Daisy', 'Canoe', 'Lycaenid butterfly', 
        'German shepherd', 'Running shoe', 'Jack-o’-lantern', 'Cellphone', 'Golf ball', 'Desktop PC', 
        'Broom', 'Pizza', 'Missile', 'Capuchin', 'Pool table', 'Mailbag', 'Convertible', 
        'Folding chair', 'Pyjama', 'Mitten', 'Electric guitar', 'Reflex camera', 'Piano', 'Mountain tent', 
        'Banana', 'Bolete', 'Digital watch', 'Elephant', 'Airliner', 'Electric locomotive', 'Radio telescope', 'Egyptian cat')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (30,20))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')
