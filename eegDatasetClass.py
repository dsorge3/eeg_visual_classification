# Imports
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import cv2
import cfg

args = cfg.parse_args()

# Dataset class
class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path):
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
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[args.time_low:args.time_high, :]

        if args.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, args.time_high - args.time_low)
        # Get label
        label = self.data[i]["label"]
        # Get image                           **MODIFICA2: RETURN IMMAGINE DAL DATASET IMAGENET**
        img_idx = self.data[i]['image']
        image = self.images[img_idx]
        # Get complete path
        dirName = image[:9]
        imgPath = "/home/d.sorge/eeg_visual_classification/datasets/imageNet/ILSVRC/Data/CLS-LOC/train/" + dirName + "/" + image + ".JPEG"
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        # Return
        return eeg, label, img
