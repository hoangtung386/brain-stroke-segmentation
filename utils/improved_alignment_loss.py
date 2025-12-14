"""
Improved Alignment Loss - FIXED for Extreme Stability

Key fixes:
1. Sobel filters with SAFE padding and clamping
2. Edge detection with gradient magnitude limiting
3. Symmetry loss with STRONG regularization
4. All operations use float32 explicitly
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedAlignmentLoss(nn.Module):
    """
    ULTRA-STABLE alignment loss
    """
    
    def __init__(self, symmetry_weight=1.0, reg_weight=0.5, edge_weight=0.3):
        super(ImprovedAlignmentLoss, self).__init__()
        
        self.symmetry_weight = symmetry_weight
        self.reg_weight = reg_weight  # Increased from 0.1
        self.edge_weight = edge_weight  # Decreased from 0.5
        
        # Sobel filters - NORMALIZED to prevent extreme values
        sobel_x = torch.FloatTensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).unsqueeze(0).unsqueeze(0) / 8.0  # Normalize by sum of absolutes
        
        sobel_y = torch.FloatTensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).unsqueeze(0).unsqueeze(0) / 8.0
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Add stability epsilon
        self.eps = 1e-6
    
    def compute_edges(self, x):
        """
        SAFE edge computation with gradient magnitude limiting
        """
        # Ensure input is in reasonable range
        x = torch.clamp(x, -10, 10)
        
        # Cast to float32 explicitly
        x = x.float()
        
        # Ensure Sobel filters match dtype and device
        sobel_x = self.sobel_x.to(dtype=x.dtype, device=x.device)
        sobel_y = self.sobel_y.to(dtype=x.dtype, device=x.device)
        
        # Apply Sobel with SAFE padding
        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)
        
        # Clamp gradients before computing magnitude
        edge_x = torch.clamp(edge_x, -5, 5)
        edge_y = torch.clamp(edge_y, -5, 5)
        
        # Compute magnitude with stability
        edges = torch.sqrt(edge_x**2 + edge_y**2 + self.eps)
        
        # Final clamping
        edges = torch.clamp(edges, 0, 10)
        
        return edges
    
    def symmetry_loss(self, aligned_slice):
        """
        STABLE symmetry loss with AGGRESSIVE regularization
        """
        # Ensure input is normalized
        aligned_slice = torch.clamp(aligned_slice, -5, 5)
        aligned_slice = aligned_slice.float()
        
        # Flip horizontally
        flipped = torch.flip(aligned_slice, dims=[-1])
        
        # Use Huber loss instead of L1 (more stable for outliers)
        diff = aligned_slice - flipped
        huber_loss = torch.where(
            torch.abs(diff) < 1.0,
            0.5 * diff**2,
            torch.abs(diff) - 0.5
        )
        sym_loss = huber_loss.mean()
        
        # Add penalty for high variance (prevents extreme alignments)
        variance_penalty = aligned_slice.var() * 0.01
        
        return sym_loss + variance_penalty
    
    def regularization_loss(self, alignment_params):
        """
        STRONG regularization to prevent extreme transformations
        """
        # L2 norm of all parameters
        angle = alignment_params[:, 0]
        shift_x = alignment_params[:, 1]
        shift_y = alignment_params[:, 2]
        
        # Quadratic penalty (stronger than linear)
        angle_loss = torch.mean(angle**2) * 10.0  # Strong penalty
        shift_loss = torch.mean(shift_x**2 + shift_y**2) * 5.0
        
        # Add penalty for large transformations
        total_transform = torch.mean(torch.abs(angle) + torch.abs(shift_x) + torch.abs(shift_y))
        extreme_penalty = torch.relu(total_transform - 0.1) * 10.0  # Penalize > 0.1
        
        reg_loss = angle_loss + shift_loss + extreme_penalty
        
        return reg_loss
    
    def edge_consistency_loss(self, aligned_slice, original_slice):
        """
        SAFE edge consistency with gradient limiting
        """
        # Ensure inputs are in safe range
        aligned_slice = torch.clamp(aligned_slice, -5, 5).float()
        original_slice = torch.clamp(original_slice, -5, 5).float()
        
        # Compute edges with safety
        edges_aligned = self.compute_edges(aligned_slice)
        edges_original = self.compute_edges(original_slice)
        
        # Use Huber loss for stability
        diff = edges_aligned - edges_original
        huber_loss = torch.where(
            torch.abs(diff) < 1.0,
            0.5 * diff**2,
            torch.abs(diff) - 0.5
        )
        
        edge_loss = huber_loss.mean()
        
        # Clamp final loss
        edge_loss = torch.clamp(edge_loss, 0, 10)
        
        return edge_loss
    
    def forward(self, aligned_slices, alignment_params_list, original_slices):
        """
        Compute total alignment loss with MAXIMUM stability
        """
        if not aligned_slices or not alignment_params_list or not original_slices:
            return torch.tensor(0.0), {}
        
        total_symmetry = 0.0
        total_regularization = 0.0
        total_edge_consistency = 0.0
        
        num_slices = len(aligned_slices)
        valid_slices = 0
        
        for aligned, params, original in zip(aligned_slices, alignment_params_list, original_slices):
            try:
                # Skip if any tensor is unhealthy
                if torch.isnan(aligned).any() or torch.isinf(aligned).any():
                    continue
                if torch.isnan(params).any() or torch.isinf(params).any():
                    continue
                if torch.isnan(original).any() or torch.isinf(original).any():
                    continue
                
                # 1. Symmetry loss
                sym_loss = self.symmetry_loss(aligned)
                if torch.isnan(sym_loss).any() or torch.isinf(sym_loss).any():
                    continue
                total_symmetry += torch.clamp(sym_loss, 0, 20)
                
                # 2. Regularization loss (STRONG)
                reg_loss = self.regularization_loss(params)
                if torch.isnan(reg_loss).any() or torch.isinf(reg_loss).any():
                    continue
                total_regularization += torch.clamp(reg_loss, 0, 20)
                
                # 3. Edge consistency loss (REDUCED weight)
                edge_loss = self.edge_consistency_loss(aligned, original)
                if torch.isnan(edge_loss).any() or torch.isinf(edge_loss).any():
                    continue
                total_edge_consistency += torch.clamp(edge_loss, 0, 20)
                
                valid_slices += 1
                
            except RuntimeError as e:
                # Skip problematic slices silently
                continue
        
        if valid_slices == 0:
            return torch.tensor(0.0), {}
        
        # Average over valid slices
        avg_symmetry = total_symmetry / valid_slices
        avg_regularization = total_regularization / valid_slices
        avg_edge_consistency = total_edge_consistency / valid_slices
        
        # Weighted combination with STRONG regularization emphasis
        total_loss = (
            self.symmetry_weight * avg_symmetry +
            self.reg_weight * avg_regularization +  # Increased weight
            self.edge_weight * avg_edge_consistency  # Decreased weight
        )
        
        # Final safety clamp
        total_loss = torch.clamp(total_loss, 0, 50)
        
        loss_dict = {
            'symmetry': avg_symmetry.item() if isinstance(avg_symmetry, torch.Tensor) else float(avg_symmetry),
            'regularization': avg_regularization.item() if isinstance(avg_regularization, torch.Tensor) else float(avg_regularization),
            'edge_consistency': avg_edge_consistency.item() if isinstance(avg_edge_consistency, torch.Tensor) else float(avg_edge_consistency),
            'total_alignment': total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss)
        }
        
        return total_loss, loss_dict


class ImprovedCombinedLoss(nn.Module):
    """
    Combined loss with STABILITY focus
    """
    
    def __init__(self, num_classes=2, dice_weight=0.5, ce_weight=0.5, 
                 alignment_weight=0.05, use_alignment=True):  # Reduced from 0.3
        super(ImprovedCombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.alignment_weight = alignment_weight  # Will be adjusted dynamically
        self.use_alignment = use_alignment
        
        # Segmentation Loss
        from monai.losses import DiceCELoss
        self.dice_ce = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=dice_weight,
            lambda_ce=ce_weight,
            smooth_nr=1e-5,  # Add smoothing
            smooth_dr=1e-5
        )
        
        # Alignment Loss with STRONG regularization
        self.alignment_loss_fn = ImprovedAlignmentLoss(
            symmetry_weight=1.0,
            reg_weight=0.5,    # Increased
            edge_weight=0.3    # Decreased
        )
    
    def forward(self, outputs, targets, aligned_slices=None, 
                alignment_params=None, original_slices=None):
        """
        Compute total loss with STABILITY checks
        """
        # Ensure float32
        outputs = outputs.float()
        
        # Clamp outputs before loss
        outputs = torch.clamp(outputs, -20, 20)
        
        # Main segmentation loss
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        
        try:
            dice_ce_loss = self.dice_ce(outputs, targets)
            
            # Check health
            if torch.isnan(dice_ce_loss).any() or torch.isinf(dice_ce_loss).any():
                print("NaN/Inf in Dice-CE loss!")
                dice_ce_loss = torch.tensor(1.0, device=outputs.device, dtype=torch.float32)
            
            # Clamp dice_ce
            dice_ce_loss = torch.clamp(dice_ce_loss, 0, 100)
            
        except RuntimeError as e:
            print(f"Error in Dice-CE: {e}")
            dice_ce_loss = torch.tensor(1.0, device=outputs.device, dtype=torch.float32)
        
        total_loss = dice_ce_loss
        alignment_loss = torch.tensor(0.0, device=outputs.device, dtype=torch.float32)
        alignment_details = {}
        
        # Add alignment loss if provided
        if (self.use_alignment and aligned_slices is not None and 
            alignment_params is not None and original_slices is not None):
            
            try:
                alignment_loss, alignment_details = self.alignment_loss_fn(
                    aligned_slices, alignment_params, original_slices
                )
                
                # Check alignment loss health
                if torch.isnan(alignment_loss).any() or torch.isinf(alignment_loss).any():
                    print("NaN/Inf in alignment loss - skipping")
                    alignment_loss = torch.tensor(0.0, device=outputs.device, dtype=torch.float32)
                else:
                    # Scale down and add
                    scaled_alignment = self.alignment_weight * alignment_loss
                    total_loss = total_loss + scaled_alignment
                
            except RuntimeError as e:
                print(f"Error in alignment loss: {e}")
                alignment_loss = torch.tensor(0.0, device=outputs.device, dtype=torch.float32)
        
        # Final safety check
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print("NaN/Inf in total loss - returning safe value")
            total_loss = torch.tensor(10.0, device=outputs.device, dtype=torch.float32)
        
        total_loss = torch.clamp(total_loss, 0, 100)
        
        return total_loss, dice_ce_loss, alignment_loss, alignment_details
        