import torch.optim as optim
from torch import nn
from torch import torch
from tqdm import tqdm
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from MyUNet.unet import UNet
from PhotoEditingDataset.PhotoEditingDataset import PhotoEditingDataset
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
 
# Resize images to nearest compatible size (e.g., divisible by 16), to solve RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 45 but got size 44 for tensor number 1 in the list.
def pad_to_divisible(image, divisor=16):
    h, w = image.shape[-2:]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    return torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))

model_name = "unet_model.pth"

def custom_collate(batch):
    # Returns list of (image, mask) without stacking
    return batch

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = UNet(in_channels=3, out_classes=1, residual=True).to(device)
    # Load saved model if it exists
    parent_dir = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
    fname = os.path.join( parent_dir, model_name )
    if os.path.isfile( fname ):
        print( f"{fname} exists, resume its training instead of creating a new.\n" )
        model.load_state_dict(torch.load(fname))


    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load dataset
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images
        transforms.ToTensor(),           # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    """
    src_dir = r"Your src image path"
    mask_dir = r"Your mask image path"

    # Split dataset into train and validation
    train_dataset = PhotoEditingDataset(
        src_dir=src_dir,
        mask_dir=mask_dir,
        # transform=transform
    )
    train_size = int(0.95 * len(train_dataset))  # 95% for training
    val_size = len(train_dataset) - train_size  # 5% for validation
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_transform = A.Compose([
        # Spatial transforms (applied to both image and mask)
        A.Rotate(limit=30, p=0.5),           # Random rotation [-30°, 30°]
        A.HorizontalFlip(p=0.5),              # 50% chance of flip
        A.VerticalFlip(p=0.5),                # 50% chance of vertical flip
        A.RandomScale(scale_limit=0.2, p=0.5), # Zoom in/out by ±20%
        A.ElasticTransform(p=0.2),            # Elastic deformations (good for medical images)
        A.GridDistortion(p=0.2),              # Distortion grid effects

        # Color transforms (applied ONLY to the image)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),

        # Normalization and tensor conversion
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats (adjust if needed)
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_dataset.transform = train_transform
    val_dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        collate_fn=custom_collate,
        batch_size=8,  # Batch size
        shuffle=True,  # Shuffle the data
        num_workers=4,  # Number of subprocesses for data loading
        pin_memory=True  # Faster data transfer to CUDA-enabled GPUs
    )
    validation_loader = DataLoader(
        val_dataset,
        collate_fn=custom_collate,
        batch_size=8,  # Can be different from train batch_size
        shuffle=False,  # Typically no need to shuffle validation data
        num_workers=4,  # For parallel data loading
        pin_memory=True  # For faster GPU transfer if using GPU
    )


    # Training loop
    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        """
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            images = images.to(device)
            masks = masks.to(device)
        """
        for batch in tqdm( train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            for images, masks in batch:
                # Forward pass
                images = pad_to_divisible( images )
                masks = pad_to_divisible( masks )
                images = images.to( device ).unsqueeze( 0 )
                masks = masks.to(device).unsqueeze( 0 )
                outputs = model( images )
                loss = criterion( outputs, masks )
                # Backward and optmize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            """
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            """

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                for images, masks in batch:
                    images = pad_to_divisible( images )
                    masks = pad_to_divisible( masks )

                    images = images.to(device).unsqueeze( 0 )
                    masks = masks.to(device).unsqueeze( 0 )
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()

        print(
            f'Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(validation_loader):.4f}')

    # Save the model
    torch.save(model.state_dict(), model_name)

