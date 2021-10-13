from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random

DATA_DIR = '/home/d.sorge/eeg_visual_classification/datasets/imageNet/ILSVRC/Data/CLS-LOC/test'
transform = ([transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor()])
X_train = datasets.ImageFolder(DATA_DIR, transform=transform)

# Dataset class
class EEGDataset(Dataset):
    
    # Constructor
    def __init__(self, eeg_signals_path, ims):
        # Initialization
        self.ims = ims
        
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
    def __getitem__(self, i, index):
        # Select sample
        image = self.ims[index]
        X = self.transform(image)
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high,:]

        if opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1,128,opt.time_high-opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label, X

batch_size = 32
eeg_dataset = EEGDataset(ims=X_train)
train_dl = DataLoader(eeg_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
# to test data loader
images, labels = next(iter(train_dl))
imshow(images[0], normalize=False)
