import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class DRAC_Loader(Dataset):
    def __init__(self, data_type, transform=None, mask = "intraretinal"):
        self.mask = mask
        # Local Variables
        self.data = 'data/Segmentation/Original/'
        self.label_loc = 'data/Segmentation/Groundtruths/'
        if mask == "intraretinal":
            self.intraretinal_loc = f'{self.label_loc}intraretinal'
        elif mask == "neovascular":
            self.neovascular_loc = f'{self.label_loc}neovascular'
        elif mask == "nonperfusion":
            self.nonperfusion_loc = f'{self.label_loc}nonperfusion'
        
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
        
        # List of number of classifications for each item.
        # Cycle through each intraretinal, neovascular, and nonperfusion data, adding +1 where there is not a 0, and adding nothing where there is.
        # self.classifications = self.NumberClassifications()

    def __getitem__(self, idx):
        # Primary image
        image = self.data_list[idx]
        
        # Blank list for the other images
        segmentations = []
        
        # Look for the other images, if not 0 at idx, append to segmentations.
        if self.mask == "intraretinal":
            segmentations.append(self.intraretinal_data[idx])
        elif self.mask == "neovascular":
            segmentations.append(self.neovascular_data[idx])
        elif self.mask == "nonperfusion":
            segmentations.append(self.nonperfusion_data[idx])
        
        # Replace any 0s with a np array of 0s, with dtype=uint8, size 1024, 1024
        segmentations = [np.zeros((1024, 1024), dtype=np.uint8) if isinstance(i, int) else i for i in segmentations]
        
        segmentation = segmentations[0]
        
        # Send to cuda:0
        segmentation = torch.tensor(segmentation, dtype=torch.float32).to('cuda:0')
        
        return image, segmentation
            
    def __len__(self):
        return len(self.items_list)
        
    def OpenImage(self, idx, location, is_idx = True):
        if is_idx:
            location = f'{location}/{self.items_list[idx]}'
            image = Image.open(location).convert('RGB')
        else:
            # Get proper location
            location = f'{location}/{idx}'
            
            # Load Item in passed location
            image = Image.open(location).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        # Convert PIL image to a Numpy Array
        if is_idx:
            image = np.array(image)
        else:
            image = np.array(image) / 255.0
        
        # Convert image to a Pytorch Tensor
        image = torch.tensor(image, dtype=torch.float32)
        
        # Send to device
        image = image.to('cuda:0')
        
        return image
    
    def NumberClassifications(self):
        classes = [1 if isinstance(self.intraretinal_data[i], int) else 0 for i in range(len(self.items_list))]
        classes = [classes[i] + 1 if isinstance(self.neovascular_data[i], int) else classes[i] for i in range(len(self.items_list))]
        classes = [classes[i] + 1 if isinstance(self.nonperfusion_data[i], int) else classes[i] for i in range(len(self.items_list))]
        return classes
    
    
    
if __name__ == '__main__':
    import os