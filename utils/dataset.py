from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import json
from torchvision import transforms
import torch

path_to_xgaze = "../Datasets/xgaze_224/"

with open(path_to_xgaze + "train_test_split.json", 'r') as file:
    data = json.load(file)
    
# print(data["train"])
def get_train_data():
    images = []
    gaze_labels = []
    # for filename in tqdm(data['train'], "Loading image and Gaze Data"):
    for filename in tqdm(data['train'], "Loading image and Gaze Data"):
        path = path_to_xgaze + "train/" + filename
        with h5py.File(path) as subject:
            # Only frontal view is required.  
            # With 18 cameras only the picture of the first
            images.append(
                torch.from_numpy(
                    subject['face_patch'][0::18]).float()/255.0)

            # Images are stored in W x H x C
            # but torch expects C x W x H
            images[-1] = images[-1].permute(0,3,2,1)
            
            gaze_labels.append(
                torch.from_numpy(
                    subject['face_gaze'][0::18]).float())

    return (torch.cat(images, dim=0), torch.cat(gaze_labels, dim=0))

def normalize_images(images):
    """
    Normalize image for ResNet to the range [0, 1]
    with: 
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]
    """
    # Normalize image for ResNet
    # with: 
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Normalize(mean,std),
    ])
    return transform(images)

def inv_normalize_images(images):
    """
    Normalize image for ResNet to the range [0, 1]
    with: 
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]
    """
    # Normalize image for ResNet
    # with: 
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std =[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std =[1, 1, 1]),
    ])
    return transform(images)

# class GazeImages(Dataset):
#     def __init__(self, normalize: bool = False):
#         self.images, self.gaze_labels, _ = get_train_data()
#         if normalize:
#             self.images = normalize_images(self.images)
#         return
#
#     def __len__(self):
#         return self.images.shape[0]
#
#     def __getitem__(self, idx):
#         image = self.images[idx, ...]
#         label = self.gaze_labels[idx, ...]
#         return image, label 

class GazeImages_forUNet(Dataset):
    def __init__(self, normalize: bool = False):
        self.images, self.gaze_labels = get_train_data()
        if normalize:
            self.images = normalize_images(self.images)
        return

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx, ...]
        label = self.gaze_labels[idx, ...]
        return image, label 

