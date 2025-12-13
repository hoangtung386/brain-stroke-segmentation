"""
Components - AlignmentNetwork with dynamic sizing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentNetwork(nn.Module):
    """
    Alignment Network - Dynamic feature size calculation
    """
    def __init__(self, input_size=(512, 512)):
        super(AlignmentNetwork, self).__init__()
        
        self.input_size = input_size
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)  # ← THÊM BatchNorm
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)  # Thêm BatchNorm
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Tính toán ĐỘNG kích thước sau pooling
        H, W = input_size
        H_after_pool = H // 4  # 2 pooling layers
        W_after_pool = W // 4
        
        flattened_size = 32 * H_after_pool * W_after_pool
        
        # Adaptive pooling để đảm bảo kích thước cố định
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # Output 16x16
        flattened_size = 32 * 16 * 16  # = 8,192
        
        self.fc1 = nn.Linear(flattened_size, 128)  # Thêm hidden layer
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 3)  # [angle, shift_x, shift_y]
        
        # Initialize to output near identity transform
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.zero_()
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W)
        Returns:
            params: (B, 3) - [angle, shift_x, shift_y]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Adaptive pooling để đảm bảo kích thước
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        params = self.fc2(x)
        
        # CLIP giá trị để tránh transformation quá lớn
        angle = torch.clamp(params[:, 0], -0.2, 0.2)  # ±11 degrees
        shift_x = torch.clamp(params[:, 1], -0.1, 0.1)
        shift_y = torch.clamp(params[:, 2], -0.1, 0.1)
        
        params = torch.stack([angle, shift_x, shift_y], dim=1)
        
        return params
    
    def get_transform_matrix(self, params, size):
        """Create affine transformation matrix from params"""
        B = params.size(0)
        angle = params[:, 0]
        shift_x = params[:, 1]
        shift_y = params[:, 2]
        
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        theta = torch.zeros(B, 2, 3, device=params.device, dtype=params.dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] = shift_x
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 1, 2] = shift_y
        
        return theta
    
    def apply_transform(self, x, params):
        """Apply transformation to input"""
        theta = self.get_transform_matrix(params, x.size())
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False, mode='bilinear')
        return x_transformed, theta


def alignment_loss(x_aligned, x_original):
    """Loss function for alignment network"""
    x_flipped = torch.flip(x_aligned, dims=[-1])
    symmetry_loss = F.l1_loss(x_aligned, x_flipped)
    return symmetry_loss


class SymmetryEnhancedAttention(nn.Module):
    """Symmetry Enhanced Attention Module"""
    
    def __init__(self, in_channels, num_partitions_h=2, num_partitions_w=2, T=1):
        super(SymmetryEnhancedAttention, self).__init__()
        
        self.C = in_channels
        self.P = num_partitions_h
        self.Q = num_partitions_w
        self.T = T
        self.d = in_channels // 2
        
        self.theta = nn.Conv2d(in_channels, self.d, 1)
        self.phi = nn.Conv2d(in_channels, self.d, 1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.h = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Thêm normalization
        self.norm = nn.LayerNorm(in_channels)
    
    def forward(self, x_slices):
        """
        x_slices: List of feature maps from adjacent slices
                  [(B, C, H, W)] * (2T+1)
        """
        center_idx = len(x_slices) // 2
        x_center = x_slices[center_idx]
        
        B, C, H, W = x_center.shape
        H_prime = H // self.P
        W_prime = W // self.Q
        
        output = torch.zeros_like(x_center)
        
        for p in range(self.P):
            for q in range(self.Q):
                h_start, h_end = p * H_prime, (p + 1) * H_prime
                w_start, w_end = q * W_prime, (q + 1) * W_prime
                
                x_partition = x_center[:, :, h_start:h_end, w_start:w_end]
                queries = self.theta(x_partition).view(B, self.d, -1)
                
                attention_sum = 0
                
                for t, x_slice in enumerate(x_slices):
                    x_slice_partition = x_slice[:, :, h_start:h_end, w_start:w_end]
                    keys = self.phi(x_slice_partition).view(B, self.d, -1)
                    values = self.g(x_slice_partition).view(B, C//2, -1)
                    
                    attn = torch.bmm(queries.transpose(1, 2), keys)
                    attn = attn / (self.d ** 0.5)
                    attn = F.softmax(attn, dim=-1)
                    out_self = torch.bmm(values, attn.transpose(1, 2))
                    
                    q_mirror = self.Q - 1 - q
                    w_start_mirror = q_mirror * W_prime
                    w_end_mirror = (q_mirror + 1) * W_prime
                    
                    x_sym_partition = x_slice[:, :, h_start:h_end, 
                                             w_start_mirror:w_end_mirror]
                    keys_sym = self.phi(x_sym_partition).view(B, self.d, -1)
                    values_sym = self.h(x_sym_partition).view(B, C//2, -1)
                    
                    attn_sym = torch.bmm(queries.transpose(1, 2), keys_sym)
                    attn_sym = attn_sym / (self.d ** 0.5)
                    attn_sym = F.softmax(attn_sym, dim=-1)
                    out_sym = torch.bmm(values_sym, attn_sym.transpose(1, 2))
                    
                    attention_sum = attention_sum + torch.cat([out_self, out_sym], dim=1)
                
                attention_sum = attention_sum.view(B, C, H_prime, W_prime)
                output[:, :, h_start:h_end, w_start:w_end] = attention_sum
        
        # Apply normalization
        output_normalized = output.permute(0, 2, 3, 1)  # (B, H, W, C)
        output_normalized = self.norm(output_normalized)
        output_normalized = output_normalized.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        output = self.out_conv(output_normalized) + x_center
        return output


class EncoderBlock3D(nn.Module):
    """3D Encoder Block with BatchNorm"""
    
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # ← THÊM Dropout để regularization
        self.dropout = nn.Dropout3d(0.1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        
        depth = x.size(2)
        if depth > 1:
            x_pooled = F.max_pool3d(x, kernel_size=2, stride=2)
        else:
            x_pooled = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        return x, x_pooled


class DecoderBlock(nn.Module):
    """2D Decoder Block"""
    
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                        kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 
                              kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
        