# Imports
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import cv2
import cfg
import numpy as np

args = cfg.parse_args()

# Dataset class
class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path, split_path, split_num=0, split_name="train"):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if args.subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                         loaded['dataset'][i]['subject'] == args.subject]
        else:
            self.data = loaded['dataset']
            self.labels = loaded["labels"]
            self.images = loaded["images"]

        # Load split
        loadedSp = torch.load(split_path)
        self.split_idx = loadedSp["splits"][split_num][split_name]
        self.split_idx = [i for i in self.split_idx if 450 <= self.data[i]["eeg"].size(1) <= 600]
        #for i in range(len(self.split_idx)):
         #   idx = self.split_idx[i]
            #self.trSet[i]["eeg"] = self.data[idx]["eeg"]
            #self.trSet[i]["label"] = self.data[idx]["label"]
            #self.trSet[i]['image'] = self.data[idx]['image']
          #  self.trSet = loaded['dataset'][idx]

        self.trSet = [loaded['dataset'][self.split_idx[i]] for i in range(len(self.split_idx))]
        #self.trSet = [self.data[self.split_idx[i]] for i in range(len(self.split_idx))]

        # Compute size
        self.size = len(self.trSet)

    def get(self):
        return len(self.trSet)

    def get2(self):
        return len(self.split_idx)

    def get3(self):
        return len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.trSet[i]["eeg"].float().t()
        eeg = eeg[args.time_low:args.time_high, :]
        # Get label
        label = self.trSet[i]["label"]
        # Get image                                         **MODIFICA2: RETURN IMMAGINE DAL DATASET IMAGENET**
        img_idx = self.trSet[i]['image']
        image = self.images[img_idx]
        # Get complete path
        dirName = image[:9]
        imgPath = "/home/d.sorge/eeg_visual_classification/datasets/imageNet/ILSVRC/Data/CLS-LOC/train/" + dirName + "/" + image + ".JPEG"
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        # Return
        return eeg, label, img
