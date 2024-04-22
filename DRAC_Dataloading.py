import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class DRAC_Loader(Dataset):
    def __init__(self, data_type, transform=None, mask = "intraretinal"):
        # Set up the mask and data type
        self.mask = mask
        self.data_type = data_type
        
        # Local Variables
        self.data = 'data/Segmentation/Original/'
        self.label_loc = 'data/Segmentation/Groundtruths/'
        
        # Retrieve the location of the different masks
        if mask == "intraretinal": self.intraretinal_loc = f'{self.label_loc}intraretinal' # String
        elif mask == "neovascular": self.neovascular_loc = f'{self.label_loc}neovascular' # String
        elif mask == "nonperfusion": self.nonperfusion_loc = f'{self.label_loc}nonperfusion' # String
        
        # Load Defaults
        self.data_loc = f'{self.data}{data_type}'
        self.transform = transform
        
        # Load Data in Location
        self.items_list = os.listdir(self.data_loc)
        
        # Load actual images into a list, using OpenImage
        self.data_list = [self.OpenImage(i, self.data_loc, is_idx = True) for i in range(len(self.items_list))]
        
        # Search for the different classifications for each image name in items-list
        if mask == "intraretinal":
            self.intraretinal_data = [self.OpenImage(i, self.intraretinal_loc, is_idx = False) if i in os.listdir(self.intraretinal_loc) else 0 for i in self.items_list]
        elif mask == "neovascular":
            self.neovascular_data = [self.OpenImage(i, self.neovascular_loc, is_idx = False) if i in os.listdir(self.neovascular_loc) else 0 for i in self.items_list]
        elif mask == "nonperfusion":
            self.nonperfusion_data = [self.OpenImage(i, self.nonperfusion_loc, is_idx = False) if i in os.listdir(self.nonperfusion_loc) else 0 for i in self.items_list]

    def __getitem__(self, idx):
        # Primary image
        image = self.OpenImage(idx, self.data_loc, is_idx = True)
        
        # Retrieve the name of the image at that index
        image_name = self.items_list[idx]
        
        # Look for the other images, if not 0 at idx, append to segmentations.
        if self.mask == "intraretinal":
            segmentation = self.intraretinal_data[idx]
        elif self.mask == "neovascular":
            segmentation = self.neovascular_data[idx]
        elif self.mask == "nonperfusion":
            segmentation = self.nonperfusion_data[idx]
        
        # If segmentation is 0, set equal to 1024x1024 numpy array of 0s
        if isinstance(segmentation, int):
            segmentation = np.zeros((1024, 1024))
        
        # If the data type is test, change the segmentation to be the name of the item
        if self.data_type == "test":
            # Change segmentation to be a string that contains the items name
            # Remove the ".png" from the end of the item name
            segmentation = self.items_list[idx][:-4]
        
        # Transforms
        if self.transform:
            image = self.transform(image)
            segmentation = self.transform(segmentation)
        
        # Convert image and segmentation to a tensor and send to device.
        image = torch.tensor(image).float().unsqueeze(0)
        segmentation = torch.tensor(segmentation).float()
        
        return image, segmentation
            
    def __len__(self):
        return len(self.items_list)
        
    def OpenImage(self, idx, location, is_idx = True):
        if is_idx:
            location = f'{location}/{self.items_list[idx]}'
            image = Image.open(location)
        else:
            # Get proper location
            location = f'{location}/{idx}'
            
            # Check if location exists
            if os.path.exists(location):
                # Load Item in passed location
                image = Image.open(location)
            else:
                return 0
        
        # Convert PIL image to a Numpy Array
        if is_idx:
            image = np.array(image)
        else:
            image = np.array(image) / 255.0
        
        return image