from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):

    def __init__(self, eeg_path, label_path, transform=None, split_name = "train"):
        self.transform = transform
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

        self.arr0 = np.random.permutation(15).tolist()
        self.arr1 = np.random.permutation(15).tolist()

    # Get size
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        from PIL import Image

        if self.y_data[i] == 0:
            # Get dir name
            dirName = "occhio_chiuso"
            if len(self.arr0) == 0:
                self.arr0 = np.random.permutation(15).tolist()
            # Get img name
            image =  "occhio_chiuso_" + str(self.arr0.pop(0))
            # Get complete path
            imgPath = "/home/d.sorge/eeg_visual_classification/dcgan/dataset_eeg_occhi/data_img/" + dirName + "/" + image + ".jpg"
        else:
            # Get dir name
            dirName = "occhio_aperto"
            if len(self.arr1) == 0:
                self.arr1 = np.random.permutation(15).tolist()
            # Get img name
            image =  "occhio_aperto_" + str(self.arr1.pop(0))
            # Get complete path
            imgPath = "/home/d.sorge/eeg_visual_classification/dcgan/dataset_eeg_occhi/data_img/" + dirName + "/" + image + ".jpg"
        
        img = Image.open(imgPath)
        img = np.asarray(img)

        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
            img = np.concatenate((img, img, img), axis=-1)

        #apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        return self.x_data[i].astype(np.float32), self.y_data[i], img
