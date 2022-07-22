# Imports
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
import torch.optim
import torch.backends.cudnn as cudnn;
cudnn.benchmark = True
import cfg
import numpy as np
import cv2
args = cfg.parse_args()

# Dataset class
class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path, split_path, split_num=0, transform=None, split_name="train"):
        self.transform = transform
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
        # **MODIFICA4: Filter data (FILTRAGGIO DATASET, IN trSet VENGONO MESSI SOLO I DATI DI TRAIN)
        self.trSet = [loaded['dataset'][self.split_idx[i]] for i in range(len(self.split_idx))]

        # Compute size
        self.size = len(self.trSet)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        from PIL import Image
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
        imgPath = "/projects/data/classification/ImageNet2012/train/" + dirName + "/" + image + ".JPEG"
        if imgPath[-4:] == ".npy":
            img = np.load(imgPath)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(imgPath)
            img = np.asarray(img)

        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
            img = np.concatenate((img, img, img), axis=-1)

        #apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # Return
        return eeg, label, img
