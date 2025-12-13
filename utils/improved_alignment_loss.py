"""
Improved Alignment Loss for SEAN Architecture

Fixes:
1. Proper symmetry loss calculation
2. Regularization to prevent trivial solutions
3. Multi-scale loss for better alignment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedAlignmentLoss(nn.Module):
    """
    Improved alignment loss with multiple components:
    1. Symmetry Loss: Penalizes asymmetry after alignment
    2. Regularization Loss: Prevents extreme transformations
    3. Edge Consistency Loss: Ensures edges are preserved
    """
    
    def __init__(self, symmetry_weight=1.0, reg_weight=0.1, edge_weight=0.5):
        super(ImprovedAlignmentLoss, self).__init__()
        
        self.symmetry_weight = symmetry_weight
        self.reg_weight = reg_weight
        self.edge_weight = edge_weight
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.FloatTensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).unsqueeze(0).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.FloatTensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).unsqueeze(0).unsqueeze(0))
    
    def compute_edges(self, x):
        """Compute edge map using Sobel filters"""
        # x: (B, 1, H, W)
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        edges = torch.sqrt(edge_x**2 + edge_y**2)
        return edges
    
    def symmetry_loss(self, aligned_slice):
        """
        Compute symmetry loss with proper scaling
        
        Args:
            aligned_slice: (B, 1, H, W) - Aligned image
        
        Returns:
            Symmetry loss value
        """
        # Flip horizontally
        flipped = torch.flip(aligned_slice, dims=[-1])
        
        # L1 loss (MAE)
        l1_loss = F.l1_loss(aligned_slice, flipped)
        
        # L2 loss (MSE) - more sensitive to large differences
        l2_loss = F.mse_loss(aligned_slice, flipped)
        
        # Combine both
        sym_loss = l1_loss + 0.5 * l2_loss
        
        return sym_loss
    
    def regularization_loss(self, alignment_params):
        """
        Regularization to prevent extreme transformations
        
        Args:
            alignment_params: (B, 3) - [angle, shift_x, shift_y]
        
        Returns:
            Regularization loss
        """
        # Penalize large angles (should be small for brain CT)
        angle_loss = torch.mean(alignment_params[:, 0]**2)
        
        # Penalize large shifts
        shift_loss = torch.mean(alignment_params[:, 1]**2 + alignment_params[:, 2]**2)
        
        # Combined regularization
        reg_loss = angle_loss + shift_loss
        
        return reg_loss
    
    def edge_consistency_loss(self, aligned_slice, original_slice):
        """
        Ensure edges are preserved after alignment
        
        Args:
            aligned_slice: (B, 1, H, W) - Aligned image
            original_slice: (B, 1, H, W) - Original image
        
        Returns:
            Edge consistency loss
        """
        # Compute edges
        edges_aligned = self.compute_edges(aligned_slice)
        edges_original = self.compute_edges(original_slice)
        
        # Ensure edges are similar
        edge_loss = F.l1_loss(edges_aligned, edges_original)
        
        return edge_loss
    
    def forward(self, aligned_slices, alignment_params_list, original_slices):
        """
        Compute total alignment loss
        
        Args:
            aligned_slices: List of (B, 1, H, W) aligned slices
            alignment_params_list: List of (B, 3) transformation parameters
            original_slices: List of (B, 1, H, W) original slices
        
        Returns:
            total_loss, loss_dict
        """
        total_symmetry = 0.0
        total_regularization = 0.0
        total_edge_consistency = 0.0
        
        num_slices = len(aligned_slices)
        
        for aligned, params, original in zip(aligned_slices, alignment_params_list, original_slices):
            # 1. Symmetry loss
            sym_loss = self.symmetry_loss(aligned)
            total_symmetry += sym_loss
            
            # 2. Regularization loss
            reg_loss = self.regularization_loss(params)
            total_regularization += reg_loss
            
            # 3. Edge consistency loss
            edge_loss = self.edge_consistency_loss(aligned, original)
            total_edge_consistency += edge_loss
        
        # Average over slices
        avg_symmetry = total_symmetry / num_slices
        avg_regularization = total_regularization / num_slices
        avg_edge_consistency = total_edge_consistency / num_slices
        
        # Weighted combination
        total_loss = (
            self.symmetry_weight * avg_symmetry +
            self.reg_weight * avg_regularization +
            self.edge_weight * avg_edge_consistency
        )
        
        loss_dict = {
            'symmetry': avg_symmetry.item(),
            'regularization': avg_regularization.item(),
            'edge_consistency': avg_edge_consistency.item(),
            'total_alignment': total_loss.item()
        }
        
        return total_loss, loss_dict


class ImprovedCombinedLoss(nn.Module):
    """
    Improved combined loss for LCNN with better alignment loss
    """
    
    def __init__(self, num_classes=2, dice_weight=0.5, ce_weight=0.5, 
                 alignment_weight=0.3, use_alignment=True):
        super(ImprovedCombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.alignment_weight = alignment_weight
        self.use_alignment = use_alignment
        
        # Segmentation Loss (Từ MONAI)
        from monai.losses import DiceCELoss
        self.dice_ce = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=dice_weight,
            lambda_ce=ce_weight
        )
        
        # Alignment Loss (Improved)
        self.alignment_loss_fn = ImprovedAlignmentLoss(
            symmetry_weight=1.0,
            reg_weight=0.1,
            edge_weight=0.5
        )
    
    def forward(self, outputs, targets, aligned_slices=None, 
                alignment_params=None, original_slices=None):
        """
        Compute total loss
        
        Args:
            outputs: (B, num_classes, H, W) - Model predictions
            targets: (B, H, W) or (B, 1, H, W) - Ground truth
            aligned_slices: List of aligned slices
            alignment_params: List of transformation parameters
            original_slices: List of original slices
        
        Returns:
            total_loss, dice_ce_loss, alignment_loss, alignment_details
        """
        # Main segmentation loss
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        
        dice_ce_loss = self.dice_ce(outputs, targets)
        
        total_loss = dice_ce_loss
        alignment_loss = torch.tensor(0.0, device=outputs.device)
        alignment_details = {}
        
        # Add improved alignment loss if slices are provided
        if (self.use_alignment and aligned_slices is not None and 
            alignment_params is not None and original_slices is not None):
            
            alignment_loss, alignment_details = self.alignment_loss_fn(
                aligned_slices, alignment_params, original_slices
            )
            
            total_loss = total_loss + self.alignment_weight * alignment_loss
        
        return total_loss, dice_ce_loss, alignment_loss, alignment_details
    
