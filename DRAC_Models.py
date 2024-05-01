import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Modify ResNet model for binary classification
class ResNetBinary(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetBinary, self).__init__()
        # Load a pre-trained ResNet model
        self.model = models.resnet18
        # Modify the last fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Convert image to torch.uint8
        x = x.type(torch.uint8)
        return self.model(x.float())