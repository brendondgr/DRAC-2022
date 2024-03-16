import torch
from torch.utils.data import Dataset
import cv2

class DRAC_Loader(Dataset):
    def __init__(self, dataloc, datalist, transform=None):
        self.dataloc = dataloc # Original Location
        self.datalist = datalist # Data Names
        self.datalocs = [f'{self.dataloc}/{g}' for g in datalist] # Data Locations

    def __getitem__(self, idx):
        return cv2.imread(self.datalocs[idx]) # Loads Image
            
    def __len__(self):
        return len(self.datalocs)
    
    def NumOfClassifications(self, idx, isIndex=True):
        def isIn(label):
            from os import listdir # Import required library
            
            label_list = ['./data/A. Segmentation/2. Groundtruths/a. Training Set/1. Intraretinal Microvascular Abnormalities',
                          './data/A. Segmentation/2. Groundtruths/a. Training Set/2. Nonperfusion Areas',
                          './data/A. Segmentation/2. Groundtruths/a. Training Set/3. Neovascularization']
        
            total = 0 # Establish Total
            for item in label_list: # Iterate through the list
                item2 = listdir(item) # Retrieve List of Items
                if label in item2: # If the label is in the list total + 1
                    total += 1
            return str(total)
        
        if isIndex:
            # Retrieve Name of Image
            name = self.datalist[int(idx)-1]
            
            return isIn(f'{name}')
        else:
            return isIn(f'{idx}')
        
if __name__ == '__main__':
    import os
    # Key Data Locations
    segmentationdata = './data/A. Segmentation/1. Original Images'
    training_set = f'{segmentationdata}/a. Training Set/'
    testing_set = f'{segmentationdata}/b. Testing Set/'

    # Label Locations
    labeldata = './data/A. Segmentation/2. Groundtruths/a. Training Set'
    ima_data = f'{labeldata}/1. Intraretinal Microvascular Abnormalities/'
    npa_data = f'{labeldata}/2. Nonperfusion Areas/'
    neo_data = f'{labeldata}/3. Neovascularization/'

    # List of Label Locations
    label_list = [ima_data, npa_data, neo_data]
    
    # Load Data
    training = os.listdir(training_set)
    testing = os.listdir(testing_set)

    # Load Labels
    ima = os.listdir(ima_data)
    npa = os.listdir(npa_data)
    neo = os.listdir(neo_data)
    
    # Create Instance
    data = DRAC_Loader(training_set, training)
    
    print(f'Total Number: {data.NumOfClassifications(7, isIndex=True)}')