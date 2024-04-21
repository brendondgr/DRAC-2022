import torch
import time

def train_model(model, train_loader, criterion, optimizer, epochs=10, criterion_name='CrossEntropyLoss', segmentation_name = "intraretinal"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.train()
    
    # Get current time
    start_time = time.time()
    
    # Losses list
    losses = []
    
    segmentations = ["intraretinal", "neovascular", "nonperfusion"]
    
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        epoch_time = time.time()
        
        for images, masks in train_loader:
            # Reorganize the images to be of shape [batch_size, channels, height, width]
            images = images.permute(0, 3, 1, 2)
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            
            # Initialize loss
            loss = 0
            
            loss = criterion(outputs, masks.long())
            
            # Take the mean of non-nan values
            scalar_loss = torch.mean(loss)
            
            # Convert to float value
            scalar_loss = scalar_loss.float()
            
            # Skips this iteration if value contained in loss tensor is nan
            #print(count)
            if torch.isnan(scalar_loss):
                continue 
            
            # Backward pass and optimize
            scalar_loss.backward()
            optimizer.step()
            
            running_loss += scalar_loss.item()
            count += 1
    
        epoch_loss = running_loss / len(train_loader)
        print(f'[Epoch {epoch + 1}/{epochs}] {segmentation_name} - Loss: {epoch_loss:.4f} | Time Spent: {time.time() - epoch_time:.2f}s')
        losses.append(epoch_loss)
    
    print(f'Training completed in {time.time() - start_time:.2f}s')
    
    return model, losses