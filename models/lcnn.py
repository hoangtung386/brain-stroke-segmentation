"""
Fixed LCNN Architecture
Properly combines local path (SEAN) and global path (ResNeXt)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sean import SEAN
from .global_path import ResNeXtGlobal


class LCNN(nn.Module):
    """
    Lightweight CNN combining local and global paths
    
    Key fixes:
    1. Proper input handling for SEAN (grayscale slices)
    2. Proper input handling for ResNeXt (RGB)
    3. Dimension alignment between paths
    """
    
    def __init__(self, num_channels=1, num_classes=2, 
                 global_impact=0.3, local_impact=0.7, T=1):
        """
        Args:
            num_channels: Number of input channels (1 for grayscale CT)
            num_classes: Number of output classes
            global_impact: Weight for global path
            local_impact: Weight for local path
            T: Number of adjacent slices for SEAN
        """
        super(LCNN, self).__init__()
        
        self.global_impact = global_impact
        self.local_impact = local_impact
        self.T = T
        self.num_classes = num_classes
        
        # Local path (SEAN) - processes grayscale slices
        self.local_path = SEAN(
            in_channels=num_channels, 
            num_classes=num_classes, 
            T=T
        )
        
        # Adapter to convert grayscale stack to RGB for global path
        # Takes center slice and replicates to 3 channels
        self.to_rgb = nn.Conv2d(num_channels, 3, kernel_size=1, bias=False)
        
        # Initialize to replicate channels
        with torch.no_grad():
            self.to_rgb.weight.fill_(1.0)
        
        # Global path (ResNeXt) - processes RGB
        self.global_path = ResNeXtGlobal(
            num_classes=num_classes,
            pretrained=False
        )
    
    def forward(self, x, return_alignment=False):
        """
        Args:
            x: (B, 2T+1, H, W) - Stack of adjacent grayscale slices
            return_alignment: If True, return alignment info from SEAN
        
        Returns:
            output: (B, num_classes, H, W)
            If return_alignment=True: (output, aligned_slices, alignment_params)
        """
        # Extract center slice for global path
        center_idx = x.size(1) // 2
        x_center = x[:, center_idx, :, :].unsqueeze(1)  # (B, 1, H, W)
        
        # Convert to RGB for global path
        x_rgb = self.to_rgb(x_center)  # (B, 3, H, W)
        
        # Global path
        global_output = self.global_path(x_rgb) * self.global_impact
        
        # Local path
        if return_alignment:
            local_output, aligned_slices, alignment_params = self.local_path(
                x, return_alignment=True
            )
            local_output = local_output * self.local_impact
            
            # Combine
            output = global_output + local_output
            
            return output, aligned_slices, alignment_params
        else:
            local_output = self.local_path(x) * self.local_impact
            
            # Combine
            output = global_output + local_output
            
            return output
    
    def get_global_output(self, x):
        """Get only global path output"""
        center_idx = x.size(1) // 2
        x_center = x[:, center_idx, :, :].unsqueeze(1)  # (B, 1, H, W)
        x_rgb = self.to_rgb(x_center)
        return self.global_path(x_rgb)
    
    def get_local_output(self, x):
        """Get only local path output"""
        return self.local_path(x)
