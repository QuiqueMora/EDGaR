from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import json
import torch

path_to_xgaze = "../Datasets/xgaze_224/"

with open(path_to_xgaze + "train_test_split.json", 'r') as file:
    data = json.load(file)
    
# print(data["train"])
def get_train_data():
    images = []
    gaze_labels = []
    for filename in tqdm(data['train'][0], "Loading image and Gaze Data"):
        path = path_to_xgaze + "train/" + filename
        with h5py.File(path) as subject:
            # Only frontal view is required.  
            # With 18 cameras only the picture of the first
            images.append(
                torch.from_numpy(
                    subject['face_patch'][0::18]))
            gaze_labels.append(
                torch.from_numpy(
                    subject['face_gaze'][0::18]))

    return (torch.cat(images, dim=0), torch.cat(gaze_labels, dim=0))

class GazeImages(Dataset):
    def __init__(self):
        self.images, self.gaze_labels = get_train_data()
        return

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx, ...]
        label = self.gaze_labels[idx, ...]
        return image, label 

