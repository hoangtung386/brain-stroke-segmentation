"""
Global path using ResNeXt50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d


class ResNeXtGlobal(nn.Module):
    """Global path using ResNeXt50 backbone"""
    
    def __init__(self, num_classes=2, pretrained=False):
        super(ResNeXtGlobal, self).__init__()
        
        # Load ResNeXt50 model
        resnext50_model = resnext50_32x4d(pretrained=pretrained)
        
        # Remove final FC layers, keep only feature extraction
        self.resnext50 = nn.Sequential(*(list(resnext50_model.children())[:-2]))
        
        # Upsampling layer
        self.upconv = nn.ConvTranspose2d(
            in_channels=2048,
            out_channels=num_classes,
            kernel_size=8,
            stride=4,
            padding=2
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        
        Returns:
            output: (B, num_classes, H, W)
        """
        x = self.resnext50(x)           # (B, 2048, H/32, W/32)
        x = self.upconv(x)              # (B, num_classes, H/8, W/8)
        x = F.interpolate(
            x, 
            scale_factor=8, 
            mode='bilinear', 
            align_corners=True
        )                               # (B, num_classes, H, W)
        return x
