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