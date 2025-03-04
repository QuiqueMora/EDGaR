from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import h5py
import json
import torch.nn.functional as FN
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import GaussianBlur, RandomCrop, ColorJitter, GaussianNoise, RandomResizedCrop
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
def dilate_mask(mask, kernel_size=3):
    """Dilate a binary mask using PyTorch."""
    # Create a dilation kernel (structuring element)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    
    # Add batch and channel dimensions if necessary
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B, 1, H, W)

    # Apply max pooling as a morphological dilation operation
    dilated_mask = FN.max_pool2d(mask.float(), kernel_size, stride=1, padding=kernel_size // 2)

    return dilated_mask.squeeze()  # Remove extra dimensions
def downsample_image(image, size=(224,224)):
    # Add batch and channel dimensions if necessary
    image = image.unsqueeze(0)
    return FN.interpolate(image, size=size, mode='bilinear', align_corners=False).squeeze(0)

class same_color_jitter():
    def __init__(self, order: torch.Tensor, brightness, contrast, saturation, hue):
        self.order = order
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        for fn_id in self.order:
            if fn_id == 0 and self.brightness is not None:
                img = F.adjust_brightness(img, self.brightness)
            elif fn_id == 1 and self.contrast is not None:
                img = F.adjust_contrast(img, self.contrast)
            elif fn_id == 2 and self.saturation is not None:
                img = F.adjust_saturation(img, self.saturation)
            elif fn_id == 3 and self.hue is not None:
                img = F.adjust_hue(img, self.hue)

        return img

class GazeImages_forUNet_pre_pro(Dataset):
    def __init__(self, normalize: bool = False, train = True, augmentation = True, heatmap = False, size = None, path_to_xgaze = "../Datasets/eth_set/"):
        # self.images, self.gaze_labels, self.subject_index = get_train_data()

        self.augmentation = augmentation
        self.heatmap = heatmap
        self.train = train
        self.size = size
        self.normalize = normalize
        self.path_to_xgaze = path_to_xgaze

        with open(self.path_to_xgaze + "train_test_split.json", 'r') as file:
            self.subject_index = json.load(file)

        ### load list of every possible combination of image-indicies within a subject
        with open(self.path_to_xgaze + "combination_list.pickle", "rb") as f:
            self.combination_list = pickle.load(f)

    def __len__(self):
        return len(self.combination_list)

    def __getitem__(self, idx):
        subject_idx, start_idx, target_idx = self.combination_list[idx]
        if self.train:
            path = self.path_to_xgaze + "train/" + self.subject_index['train'][subject_idx]

            with h5py.File(path) as subject:
                # Only frontal view is required.  
                # With 18 cameras only the picture of the first
                imgages = subject['face_patch']
                gazes = subject['face_gaze']
                # combine eye masks
                mask = torch.from_numpy(subject['left_eye_mask'][target_idx]) + torch.from_numpy(subject['right_eye_mask'][target_idx])
                image_start = torch.from_numpy(imgages[start_idx]).float()/255.0
                label_start = torch.from_numpy(gazes[start_idx]).float()
                image_target = torch.from_numpy(imgages[target_idx]).float()/255.0
                label_target = torch.from_numpy(gazes[target_idx]).float()

        # Images are stored in W x H x C
        # but torch expects C x W x H
        image_start = image_start.permute(2,0,1)
        image_target = image_target.permute(2,0,1)
        
        # dateset augmentation
        if self.augmentation:
        # combine images, along channels in order to have the same transform
            combined = torch.cat((image_start, image_target, mask.expand(1, -1, -1)), dim=0)
            # spacial augmentation
            # crop = RandomCrop(size=432)(combined)
            crop = RandomResizedCrop(size=224, ratio=[1,1], scale=[.35,1.0])(combined)

            # color augmentation TODO augmentation for brightness
            generator = ColorJitter(brightness=0.0, hue=.3)
            color = same_color_jitter(*ColorJitter.get_params(generator.brightness, generator.contrast, generator.saturation,generator.hue))

            image_start = color(crop[0:3,...])
            image_target = color(crop[3:6,...])
            mask = crop[6:, ...]
            # augmentation for camera noise
            noise = torch.rand_like(image_start)*.1
            image_start = torch.clamp(image_start + noise, 0, 1)
            image_target = torch.clamp(image_target + noise, 0, 1)
            #todo maby make smaller for faster training
            # image_start = downsample_image(image_start, size=(224,224))
            # image_target = downsample_image(image_target, size=(224,224))
            # mask = downsample_image(mask, size=(224,224))

        if self.heatmap:
            # todo: großmachen (erode/dilate)
            # Todo dann bluren mit normalisiertem kernel
            mask = dilate_mask(mask.int(),kernel_size=19).float()
            mask = mask.unsqueeze(0)
            mask = GaussianBlur(kernel_size=39, sigma = 9)(mask.float())
            mask /= torch.max(mask)


        # Normalize images
        if self.normalize:
            image_start = normalize_images(image_start)
            image_target = normalize_images(image_target)

        return image_start, label_start, image_target, label_target, mask

class GazeImages_forUNet(Dataset):
    def __init__(self, normalize: bool = False, train = True):
        # self.images, self.gaze_labels, self.subject_index = get_train_data()

        self.train = train
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
        if self.train:
            path = self.path_to_xgaze + "train/" + self.subject_index['train'][subject_idx]

            with h5py.File(path) as subject:
                # Only frontal view is required.  
                # With 18 cameras only the picture of the first
                imgages = subject['face_patch']
                gazes = subject['face_gaze']
                image_start = torch.from_numpy(imgages[18*start_idx]).float()/255.0
                label_start = torch.from_numpy(gazes[18*start_idx]).float()
                image_target = torch.from_numpy(imgages[18*target_idx]).float()/255.0
                label_target = torch.from_numpy(gazes[18*target_idx]).float()

        else:
            path = self.path_to_xgaze + "test/" + self.subject_index['test'][subject_idx]

            with h5py.File(path) as subject:
                # Only frontal view is required.  
                # With 18 cameras only the picture of the first
                imgages = subject['face_patch']
                gazes = subject['face_gaze']
                image_start = torch.from_numpy(imgages[18*start_idx]).float()/255.0
                label_start = torch.from_numpy(gazes[18*start_idx]).float()
                image_target = torch.from_numpy(imgages[18*target_idx]).float()/255.0
                label_target = torch.from_numpy(gazes[18*target_idx]).float()

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

    combination_list = []
    for idx, subject in enumerate(tqdm(subjects)): 
        with h5py.File(subject) as s:
            # Only frontal view is required.
            # e.g. cameras with index == 0
            a = [i for i, c in enumerate(s['cam_index']) if c == 0]
        combination_list.extend([(idx,) + t for t in combinations_with_replacement(a, 2)])

    with open("combination_list.pickle", "wb") as f:
        pickle.dump(combination_list, f)
