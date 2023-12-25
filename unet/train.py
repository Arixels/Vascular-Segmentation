# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import u_net

# Assuming you have a custom dataset class. Adjust the import and instantiation accordingly.
# from your_dataset_module import YourDataset

# Function to display images and segmentation masks during training
def display_images(images, masks, predictions):
    for i in range(images.shape[0]):
        plt.subplot(1, 3, 1)
        plt.imshow(images[i, 0, :, :], cmap='gray')
        plt.title('Input Image')

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i, 0, :, :], cmap='gray')
        plt.title('Ground Truth Mask')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i, 0, :, :], cmap='gray')
        plt.title('Predicted Mask')

        plt.show()

# Initialize your U-Net model
model = u_net.UNet()

# Define loss function (e.g., CrossEntropyLoss for multi-class segmentation or BCEWithLogitsLoss for binary)
criterion = nn.CrossEntropyLoss()

# Define optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=0.0004)

# Number of training epochs
num_epochs = 1

# Loss threshold for checkpointing
loss_threshold = 0.0004

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        output = model(data)

        # Compute the loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Print progress and display images
        if batch_idx % your_print_interval == 0:
            print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
            display_images(data.numpy(), target.numpy(), torch.argmax(output, dim=1).detach().numpy())

    # Calculate average loss for the epoch
    avg_loss = running_loss / len(train_loader)

    # Print average loss for the epoch
    print(f"Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss}")

    # Save model checkpoints if the average loss is below the threshold
    if avg_loss < loss_threshold:
        checkpoint_path = f'unet_checkpoint_epoch_{epoch}_loss_{avg_loss}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)

# Save the final trained model
torch.save(model.state_dict(), 'unet_segmentation_model.pth')

# %%
