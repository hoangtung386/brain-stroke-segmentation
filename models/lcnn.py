"""
LCNN - Architecture
Combines local path (SEAN) and global path (ResNeXt)
"""
import torch
import torch.nn as nn
from .sean import SEAN
from .global_path import ResNeXtGlobal


class LCNN(nn.Module):
    """
    Combines local and global paths
    """
    
    def __init__(self, num_channels=3, num_classes=2, 
                 global_impact=0.3, local_impact=0.7, T=1):
        """
        Args:
            num_channels: Number of input channels
            num_classes: Number of output classes
            global_impact: Weight for global path
            local_impact: Weight for local path
            T: Number of adjacent slices for SEAN
        """
        super(LCNN, self).__init__()
        
        self.global_impact = global_impact
        self.local_impact = local_impact
        
        # Local path (SEAN)
        self.local_path = SEAN(
            in_channels=num_channels, 
            num_classes=num_classes, 
            T=T
        )
        
        # Global path (ResNeXt)
        self.global_path = ResNeXtGlobal(
            num_classes=num_classes,
            pretrained=False
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) for global path
               or (B, 2T+1, H, W) for local path
        
        Returns:
            output: (B, num_classes, H, W)
        """
        # Global path
        x_global = self.global_path(x) * self.global_impact
        
        # Local path
        x_local = self.local_path(x) * self.local_impact
        
        # Combine
        output = x_global + x_local
        
        return output
    
    def get_global_output(self, x):
        """Get only global path output"""
        return self.global_path(x)
    
    def get_local_output(self, x):
        """Get only local path output"""
        return self.local_path(x)
