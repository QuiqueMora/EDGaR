from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import h5py
import json
from torchvision import transforms
from itertools import combinations_with_replacement
import torch

    
# print(data["train"])
# def get_train_data(combination_tuple):
#     """
#     takes in combination tuple in form:
#     (subject_index, start_index, target_index)
#     and outputs the tuple
#     (start_image, start_lable, target_image, target_lable)
#     """
#     # list of where each subject begins in the dataset
#     for filename in tqdm(subject_index['train'], "Loading image and Gaze Data"):
#         path = path_to_xgaze + "train/" + filename
#         with h5py.File(path) as subject:
#             # Only frontal view is required.  
#             # With 18 cameras only the picture of the first
#             current_images = torch.from_numpy(subject['face_patch'][0::18]).float()/255.0
#             # print(current_images.shape)
#             # images.append(current_images)
#
#             # Images are stored in W x H x C
#             # but torch expects C x W x H
#             # images[-1] = images[-1].permute(0,3,1,2)
#             
#             # gaze_labels.append(
#             #     torch.from_numpy(
#             #         subject['face_gaze'][0::18]).float())
#             #
#         the_count += current_images.shape[0]
#         subject_index.append(the_count)
#         count += 1
#         # if count == 2:
#         #     break
#
#
#     return subject_index
#     # return (torch.cat(images, dim=0), torch.cat(gaze_labels, dim=0), subject_index)

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
        # self.images, self.gaze_labels, self.subject_index = get_train_data()

        self.normalize = normalize
        self.path_to_xgaze = "../Datasets/xgaze_224/"

        with open(self.path_to_xgaze + "train_test_split.json", 'r') as file:
            self.subject_index = json.load(file)

        ### load list of every possible combination of image-indicies within a subject
        with open(self.path_to_xgaze + "combination_list.pickle", "rb") as f:
            self.combination_list = pickle.load(f)


    def __len__(self):
        return len(self.combination_list)

    def __getitem__(self, idx):
        subject_idx, start_idx, target_idx = self.combination_list[idx]
        path = self.path_to_xgaze + "train/" + self.subject_index['train'][subject_idx]

        with h5py.File(path) as subject:
            # Only frontal view is required.  
            # With 18 cameras only the picture of the first
            imgages = subject['face_patch']
            gazes = subject['face_gaze']
            image_start = torch.from_numpy(imgages[0::18][start_idx]).float()/255.0
            label_start = torch.from_numpy(gazes[0::18][start_idx]).float()
            image_target = torch.from_numpy(imgages[0::18][target_idx]).float()/255.0
            label_target = torch.from_numpy(gazes[0::18][target_idx]).float()

        # Images are stored in W x H x C
        # but torch expects C x W x H
        image_start = image_start.permute(2,0,1)
        image_target = image_target.permute(2,0,1)
        
        # Normalize images
        if self.normalize:
            image_start = normalize_images(image_start)
            image_target = normalize_images(image_target)

        return image_start, label_start, image_target, label_target 
if __name__ == "__main__":
    with open("subject_indices.json", 'r') as file:
        subject_index = json.load(file)

    combination_list = []

    for i in range(len(subject_index)-1):
        a = range(subject_index[i + 1] - subject_index[i])
        combination_list.extend([(i,) + t for t in combinations_with_replacement(a, 2)])

    with open("combination_list.pickle", "wb") as f:
        pickle.dump(combination_list, f)

