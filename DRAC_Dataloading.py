from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import os
import torch
from torchvision import transforms


class DRAC_Loader(Dataset):
    def __init__(self, data = [], data_names = [], labels_loc = [], label_names = [], data_type = "train", mask = "intraretinal", transform=None, rotation = False):
        # Intialize the basic variables
        self.data_loc = data
        self.image_names = data_names
        self.labels_loc = labels_loc
        self.label_names = label_names
        self.data_type = data_type
        self.mask = mask
        self.transform = transform
        self.rotation = rotation
        
        # Load the data
        self.images = self.load_data(self.data_loc)
        self.labels = self.load_data(self.labels_loc)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]/255.0
        image_name = self.image_names[idx]
        
        # Look for image_name in label_names, if does not exist, return None
        try:
            label_idx = self.label_names.index(image_name)
            label = self.labels[label_idx]
            #print(f"Found Label at index {label_idx} for {image_name}")
        except:
            label = np.zeros((1024, 1024))
        
        # Apply the transformations
        if self.transform:
            # Apply a ToTensor transformation
            image = self.transform(image)
            label = self.transform(label)
            
        # If rotation is True, apply random rotation, 90, 180, 270 degrees
        if self.rotation:
            rotation = np.random.randint(0, 4)
            image = np.rot90(image, rotation)
            label = np.rot90(label, rotation)
        
        return image, label
    
    def load_data(self, data):
        # Load data from "data" list, which is a list of paths to the images
        images = []
        
        for file in data:
            image = sitk.ReadImage(file)
            image = sitk.GetArrayFromImage(image)
            images.append(image)
        
        return images