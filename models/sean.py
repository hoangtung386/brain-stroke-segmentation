"""
SEAN (Symmetry Enhanced Attention Network) architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import (
    AlignmentNetwork, 
    SymmetryEnhancedAttention,
    EncoderBlock3D, 
    DecoderBlock
)


class SEAN(nn.Module):
    """
    Proper gradient flow and initialization
    """
    
    def __init__(self, in_channels=1, num_classes=2, T=1, input_size=(512, 512)):
        super(SEAN, self).__init__()
        
        self.T = T
        
        # Alignment Network với input size động
        self.alignment_net = AlignmentNetwork(input_size=input_size)
        
        # HybridUNet Encoder (3D)
        self.enc1 = EncoderBlock3D(1, 64)
        self.enc2 = EncoderBlock3D(64, 128)
        self.enc3 = EncoderBlock3D(128, 256)
        self.enc4 = EncoderBlock3D(256, 512)
        
        # Bottleneck với Dropout
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),  # ← THÊM Dropout
            nn.Conv3d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Symmetry Enhanced Attention
        self.sea = SymmetryEnhancedAttention(
            1024, 
            num_partitions_h=4,
            num_partitions_w=4, 
            T=T
        )
        
        # Decoder (2D)
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        
        # Final output
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Proper weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights to prevent gradient explosion
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_alignment=False):
        """
        Args:
            x: (B, 2T+1, H, W) - Stack of adjacent slices
            return_alignment: If True, return alignment info
        
        Returns:
            output: (B, num_classes, H, W)
            If return_alignment=True: (output, aligned_slices, alignment_params)
        """
        B, num_slices, H, W = x.shape
        center_idx = num_slices // 2
        
        # 1. Alignment - align each slice với gradient scaling
        aligned_slices = []
        alignment_params = []
        
        for i in range(num_slices):
            slice_i = x[:, i:i+1, :, :]  # (B, 1, H, W)
            
            # Thêm gradient scaling để tránh explosion
            with torch.cuda.amp.autocast(enabled=False):
                slice_i_fp32 = slice_i.float()
                params = self.alignment_net(slice_i_fp32)
                aligned_slice, theta = self.alignment_net.apply_transform(slice_i_fp32, params)
            
            aligned_slices.append(aligned_slice)
            alignment_params.append(params)
        
        # 2. Stack aligned slices into 3D volume
        x_aligned = torch.stack(aligned_slices, dim=2)  # (B, 1, 2T+1, H, W)
        
        # 3. 3D Encoding
        # Use gradient checkpointing for encoder to save memory
        from torch.utils.checkpoint import checkpoint
        
        def encoder_forward(x_in):
            s1, x_out = self.enc1(x_in)
            s2, x_out = self.enc2(x_out)
            s3, x_out = self.enc3(x_out)
            s4, x_out = self.enc4(x_out)
            return s1, s2, s3, s4, x_out

        if self.training and x_aligned.requires_grad:
            skip1, skip2, skip3, skip4, x = checkpoint(
                encoder_forward, x_aligned, use_reentrant=False
            )
        else:
            skip1, skip2, skip3, skip4, x = encoder_forward(x_aligned)
        
        # 4. Bottleneck
        x = self.bottleneck(x)
        
        # 5. Extract middle slice and apply SEA
        mid_depth = x.size(2) // 2
        
        sea_inputs = []
        for d in range(max(0, mid_depth - self.T),
                      min(x.size(2), mid_depth + self.T + 1)):
            sea_inputs.append(x[:, :, d, :, :])
        
        x_sea = self.sea(sea_inputs)
        
        # 6. 2D Decoding (middle slice only)
        skip4_mid = skip4[:, :, skip4.size(2)//2, :, :]
        skip3_mid = skip3[:, :, skip3.size(2)//2, :, :]
        skip2_mid = skip2[:, :, skip2.size(2)//2, :, :]
        skip1_mid = skip1[:, :, skip1.size(2)//2, :, :]
        
        x = self.dec4(x_sea, skip4_mid)
        x = self.dec3(x, skip3_mid)
        x = self.dec2(x, skip2_mid)
        x = self.dec1(x, skip1_mid)
        
        # 7. Final segmentation
        output = self.final(x)
        
        if return_alignment:
            return output, aligned_slices, alignment_params
        
        return output
        