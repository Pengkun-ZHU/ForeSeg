import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import rawpy
import os
import torchvision.transforms as transforms

class PhotoEditingDataset( Dataset ):
    def __init__( self, src_dir, mask_dir, transform=None ):
        self.src_dir = src_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.src_images = os.listdir( src_dir )
        self.mask_images = os.listdir( mask_dir )
        assert len( self.src_images ) == len( self.mask_images ),\
                f"Size mismatched in {src_dir} vs {mask_dir}.\n" 
        #TODO: ensure same suffix for raw and edited, respectively, so as to ensure their order is well aligned.

    def __len__( self ):
        return len( self.src_images )

    def __getitem__( self, idx ):
        src_path = os.path.join( self.src_dir, self.src_images[idx] )
        mask_path = os.path.join( self.mask_dir, self.mask_images[idx] )
        #TODO: support grayscale to tradeoff memory overhead with quality.

        def _raw_to_RGB( raw_path ):
            with rawpy.imread( raw_path ) as raw:
                rgb = raw.postprocess() # N.B. If needed, pass output_bps=16 into postprocess to get 16 bits output, and divide by 511 for normalize.
            return rgb.astype( np.float32 ) / 255.0 # Normalize to [0, 1] for faster convergence.

        def _Image_to_RGB( image_path ):
            # Load image (always normalize)
            image = Image.open( image_path ).convert("RGB")
            image = transforms.ToTensor()(image)  # [0,1] range + [C,H,W]
            # image = transforms.Resize( ( 256, 256 ) )(image)
            return image

        def _Image_to_Grayscale( image_path ):
            # Load mask (always binarize)
            mask = Image.open( image_path ).convert( "L" )
            mask = transforms.ToTensor()( mask )  # [1,H,W] in [0,1] (from 0-255)
            mask = ( mask > 0.5 ).float()  # Always binarize (critical step!)
            # mask = transforms.Resize( (256, 256) )(mask)
            return mask

        src_image = _Image_to_RGB( src_path )
        mask_image = _Image_to_Grayscale( mask_path )

        if self.transform:
            augmented = self.transform( image=src_image, mask=mask_image )
            src_image = augmented["image"]
            mask_image = augmented["mask"]
        else:
            src_image = transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )( src_image )

        return src_image, mask_image

"""
# Test
src_dir = r"E:\ppr10k\source_360p"
mask_dir = r"E:\ppr10k\masks_360p"


src, mask = PhotoEditingDataset( src_dir, mask_dir ).__getitem__(0)

print( f"src.shape is {src.shape}" )
print( f"{src}" )
print( f"mask.shape is {mask.shape}" )
print( f"{mask}" )
"""
