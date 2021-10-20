# Define options
import argparse

parser = argparse.ArgumentParser(description="Template")
# Dataset options

# Data - Data needs to be pre-filtered and filtered data is available

# Model type/options
parser.add_argument('-mt', '--model_type', default='lstm',
                    help='specify which generator should be used: lstm|EEGChannelNet')
# It is possible to test out multiple deep classifiers:
# - lstm is the model described in the paper "Deep Learning Human Mind for Automated Visual Classification”, in CVPR 2017
# - model10 is the model described in the paper "Decoding brain representations by multimodal learning of neural activity and visual features", TPAMI 2020
parser.add_argument('-mp', '--model_params', default='', nargs='*', help='list of key=value pairs of model options')
parser.add_argument('--pretrained_net', default='', help="path to pre-trained net (to continue training)")

# Training options
parser.add_argument("-b", "--batch_size", default=16, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=200, type=int, help="training epochs")

# Save options
parser.add_argument('-sc', '--saveCheck', default=100, type=int, help="learning rate")

# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")

# Parse arguments
opt = parser.parse_args()
print(opt)

# Imports
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import cv2

# Dataset class
class EEGDataset:

    # Constructor
    def __init__(self, args, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if args.subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                         loaded['dataset'][i]['subject'] == args.subject]
        else:
            self.data = loaded['dataset']
            self.labels = loaded["labels"]
            self.images = loaded["images"]

        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, args, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[args.time_low:args.time_high, :]

        if opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, args.time_high - args.time_low)
        # Get label
        label = self.data[i]["label"]
        # Get image                           **MODIFICA2: RETURN IMMAGINE DAL DATASET IMAGENET**
        image = self.images[i]
        # Get complete path
        dirName = image[:9]
        imgPath = "/home/d.sorge/eeg_visual_classification/datasets/imageNet/ILSVRC/Data/CLS-LOC/train/" + dirName + "/" + image + ".JPEG"
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        # Return
        return eeg, label, img
