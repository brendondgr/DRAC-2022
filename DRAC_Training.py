import torch
import time
import collections

def train_model(model, train_loader, criterion, optimizer, epochs=10, criterion_name='CrossEntropyLoss', segmentation_name = "intraretinal"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.train()
    
    # Get current time
    start_time = time.time()
    
    # Losses list
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        epoch_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            '''import matplotlib.pyplot as plt
            
            # Plot both inputs and targets on the same figure, separate plots
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(inputs[0, 0, :, :].cpu().numpy())
            axs[0].set_title('Input')
            axs[1].imshow(targets[0, :, :].cpu().numpy())
            axs[1].set_title('Target')
            
            # Save this plot as a png in "graphs" folder as the batch_idx-th image
            plt.savefig(f'graphs/{batch_idx}.png')  '''          
            
            # Move data to the appropriate device (CPU or GPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute the model output
            outputs = model(inputs)
            
            if isinstance(outputs, collections.OrderedDict):
                # Assuming the output tensor is stored under the key 'out'
                outputs = outputs['out']

            # Calculate the loss
            loss = criterion(outputs, targets.long())
            
            # Set the loss gradient to True
            loss.requires_grad = True

            # Backward pass: compute the gradient of the loss with respect to model parameters
            loss.backward()

            # Update the model weights
            optimizer.step()

            # Accumulate the loss and calculate accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
    
        # Calculate avereage loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
    
        print(f'[Epoch {epoch + 1}/{epochs}] {segmentation_name} - Loss: {epoch_loss:.4f} | Time Spent: {time.time() - epoch_time:.2f}s')
        losses.append(epoch_loss)
    
    print(f'Training completed in {time.time() - start_time:.2f}s')
    
    return model, losses