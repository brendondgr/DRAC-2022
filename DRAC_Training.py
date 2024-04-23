import torch
import time
import collections
import resource
import GPUtil
from GPUtil import showUtilization as gpu_usage

def print_gpu_memory_usage(when_use):
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU: {gpu.name} @ {when_use}")
        print(f"Total VRAM: {gpu.memoryTotal / 1024:.2f} GB")
        print(f"Used VRAM: {gpu.memoryUsed / 1024:.2f} GB")
        print(f"Free VRAM: {gpu.memoryFree / 1024:.2f} GB")
        print(f"GPU Load: {gpu.load * 100:.2f}%\n")

def current_memory_usage():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

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
        
        print_gpu_memory_usage("Pre - Before Train Loop")
        #before_memory = current_memory_usage()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #after_memory = current_memory_usage()
            #print(f"Memory when items are loaded: {after_memory - before_memory} MB")
            print_gpu_memory_usage("(1) After Memory Items are loaded in")
            '''import matplotlib.pyplot as plt
            
            # Plot both inputs and targets on the same figure, separate plots
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(inputs[0, 0, :, :].cpu().numpy())
            axs[0].set_title('Input')
            axs[1].imshow(targets[0, :, :].cpu().numpy())
            axs[1].set_title('Target')
            
            # Save this plot as a png in "graphs" folder as the batch_idx-th image
            plt.savefig(f'graphs/{batch_idx}.png')'''
            
            # Move data to the appropriate device (CPU or GPU)
            inputs, targets = inputs.to(device), targets.to(device)
            
            print_gpu_memory_usage("(2) After being passed to device")
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            print_gpu_memory_usage("(3) After optimizer is zeroed")
            
            # Forward pass: compute the model output
            outputs = model(inputs)
            
            print_gpu_memory_usage("(3.5) After model is passed through inputs")
            
            if isinstance(outputs, collections.OrderedDict):
                # Assuming the output tensor is stored under the key 'out'
                outputs = outputs['out']

            print_gpu_memory_usage("(4) After model is passed through inputs and outputs are stored in 'out'")
            
            # Calculate the loss
            loss = criterion(outputs, targets.to(torch.int8))
            
            print_gpu_memory_usage("(5) After loss is calculated")
            
            # Set the loss gradient to True
            loss.requires_grad = True
            
            print_gpu_memory_usage("(6) After loss requires_grad is set to True")

            # Backward pass: compute the gradient of the loss with respect to model parameters
            loss.backward()
            
            print_gpu_memory_usage("(7) After loss is backward-ed")

            # Update the model weights
            optimizer.step()
            
            print_gpu_memory_usage("(8) After optimizer is stepped")

            # Accumulate the loss and calculate accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            print_gpu_memory_usage("(9) After loss is accumulated and accuracy is calculated")
    
        # Calculate avereage loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
    
        print(f'[Epoch {epoch + 1}/{epochs}] {segmentation_name} - Loss: {epoch_loss:.4f} | Time Spent: {time.time() - epoch_time:.2f}s')
        losses.append(epoch_loss)
    
    print(f'Training completed in {time.time() - start_time:.2f}s')
    
    return model, losses