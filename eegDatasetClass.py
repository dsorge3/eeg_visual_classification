# Define options
import argparse

parser = argparse.ArgumentParser(description="Template")
# Dataset options

# Data - Data needs to be pre-filtered and filtered data is available

### BLOCK DESIGN ###
# Data
# parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_55_95_std.pth", help="EEG dataset path") #55-95Hz
parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_5_95_std.pth", help="EEG dataset path")  # 5-95Hz
# parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_14_70_std.pth", help="EEG dataset path") #14-70Hz
# Splits
parser.add_argument('-sp', '--splits-path', default=r"data\block\block_splits_by_image_all.pth",
                    help="splits path")  # All subjects
# parser.add_argument('-sp', '--splits-path', default=r"data\block\block_splits_by_image_single.pth", help="splits path") #Single subject
### BLOCK DESIGN ###

parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number")  # leave this always to zero.

# Subject selecting
parser.add_argument('-sub', '--subject', default=0, type=int,
                    help="choose a subject from 1 to 6, default is 0 (all subjects)")

# Time options: select from 20 to 460 samples from EEG data
parser.add_argument('-tl', '--time_low', default=20, type=float, help="lowest time value")
parser.add_argument('-th', '--time_high', default=460, type=float, help="highest time value")

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
import torch;

torch.utils.backcompat.broadcast_warning.enabled = True
import torch.optim
import torch.backends.cudnn as cudnn;

cudnn.benchmark = True
import cv2

# Dataset class
class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if opt.subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                         loaded['dataset'][i]['subject'] == opt.subject]
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
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high, :]

        if opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, opt.time_high - opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Get image
        image = self.images[i]
        # Get complete path
        dirName = image[:9]
        imgPath = "/home/d.sorge/eeg_visual_classification/datasets/imageNet/ILSVRC/Data/CLS-LOC/train/" + dirName + "/" + image + ".JPEG"
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        # Return
        return eeg, label, img

# Load dataset
dataset = EEGDataset(opt.eeg_dataset)
# testing
eeg,label,img = dataset.__getitem__(0)
print(img)

#print("Nome immagine:", imgName)
#dirName = imgName[:9]
#print("Nome directory:", dirName)
#data_dir = "/home/d.sorge/eeg_visual_classification/datasets/imageNet/ILSVRC/Data/CLS-LOC/train/" + dirName + "/" + imgName + ".JPEG"
#print("Path completo:", data_dir)
#img = cv2.imread(data_dir, cv2.IMREAD_COLOR)
#print(img)
