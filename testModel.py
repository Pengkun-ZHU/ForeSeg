import torch
import torch.nn as nn
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from unet.unet.unet import UNet
from PIL import Image
from torchvision import transforms

from torchvision.transforms import ToPILImage
# import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_classes=1, residual=True ).to(device)
model.load_state_dict( torch.load( 'unet_model.pth' ) )
model.eval()

# image = Image.open(r"E:\dataset\edited\a0013-MB_20030906_001.tif").convert("RGB")
image = Image.open(r"E:\ppr10k\source_360p\44_4.tif").convert("RGB")

# Define the target height (360p)
target_height = 360

# Calculate the new width while maintaining aspect ratio
width, height = image.size
aspect_ratio = width / height
target_width = int(target_height * aspect_ratio)

# Apply resizing
resize_transform = transforms.Resize((target_height, target_width))
# image = resize_transform(image)

# Continue with ToTensor, Normalize, etc.
def pad_to_divisible(image, divisor=16):
    h, w = image.shape[-2:]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    return torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))
image = transforms.ToTensor()(image)
# image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
image = pad_to_divisible(image)  # Your existing padding function

to_pil = ToPILImage()
to_pil( image.squeeze(0) ).show()

with torch.no_grad():
    image = image.to(device).unsqueeze( 0 )
    output = model(image)
    probabilities = torch.sigmoid( output )  # Convert to [0, 1]
    mask = ( probabilities > 0.5 ).float()  # Threshold at 0.5
    # print( mask )
    # image = mask
    image = mask * image
    image = to_pil( image.squeeze(0) )

    image.show()
