import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, masks):
        # Reshape the outputs to (batch_size, num_classes, height, width)
        outputs = outputs.reshape(outputs.size(0), outputs.size(1), outputs.size(2), outputs.size(3))
        
        # Apply softmax to the outputs
        outputs = F.softmax(outputs, dim=1)

        # Flatten the masks
        masks_flat = [mask.view(-1) for mask in masks]

        # Compute Dice Similarity Coefficient for each class
        dice_loss = 0
        for i, mask_flat in enumerate(masks_flat):
            outputs_flat = outputs[:, i, :, :].contiguous().view(-1)
            intersection = torch.sum(outputs_flat * mask_flat)
            union = torch.sum(outputs_flat) + torch.sum(mask_flat)
            dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice_coeff

        # Average the dice loss across all classes
        dice_loss /= len(masks)

        return dice_loss
