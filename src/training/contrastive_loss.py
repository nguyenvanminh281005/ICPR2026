"""Contrastive learning and combined loss functions for multi-frame LPR."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFrameContrastiveLoss(nn.Module):
    """
    Contrastive loss for multi-frame learning.
    Enforces that frames from the same track should have similar features.
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent contrastive loss.
        
        Args:
            frame_features: Features from T frames of B samples [B, T, D]
                           Frames from same sample are positive pairs.
        
        Returns:
            Contrastive loss value
        """
        B, T, D = frame_features.shape
        
        # Normalize features
        frame_features = F.normalize(frame_features, dim=-1)
        
        # Reshape to [B*T, D]
        features = frame_features.view(B * T, D)
        
        # Compute similarity matrix: [B*T, B*T]
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive mask: frames from same sample are positive
        # Labels: [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, ...] for T=5
        labels = torch.arange(B, device=features.device).repeat_interleave(T)
        
        # Create mask for positive pairs (same sample, different frame)
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Mask out self-similarity (diagonal)
        self_mask = torch.eye(B * T, dtype=torch.bool, device=features.device)
        positive_mask = positive_mask.masked_fill(self_mask, 0)
        
        # Mask out self from similarity matrix
        sim_matrix = sim_matrix.masked_fill(self_mask, float('-inf'))
        
        # For each sample, compute log_softmax and select positive pairs
        log_prob = F.log_softmax(sim_matrix, dim=1)
        
        # Mean log probability of positive pairs
        num_positives = positive_mask.sum(dim=1).clamp(min=1)
        loss = -(positive_mask * log_prob).sum(dim=1) / num_positives
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SequenceContrastiveLoss(nn.Module):
    """
    Contrastive loss at sequence level.
    Samples with same label should have similar sequence features.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        sequence_features: torch.Tensor,
        labels: list
    ) -> torch.Tensor:
        """
        Args:
            sequence_features: Sequence features [B, D] (e.g., after pooling)
            labels: List of string labels for each sample
        
        Returns:
            Contrastive loss
        """
        B = sequence_features.size(0)
        device = sequence_features.device
        
        # Normalize
        features = F.normalize(sequence_features, dim=-1)
        
        # Compute similarity
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create label mask (samples with same label are positive)
        label_matrix = torch.zeros(B, B, device=device)
        for i in range(B):
            for j in range(B):
                if i != j and labels[i] == labels[j]:
                    label_matrix[i, j] = 1.0
        
        # If no positive pairs in batch, return 0
        if label_matrix.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Mask diagonal
        mask = torch.eye(B, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Compute loss
        log_prob = F.log_softmax(sim_matrix, dim=1)
        num_positives = label_matrix.sum(dim=1).clamp(min=1)
        loss = -(label_matrix * log_prob).sum(dim=1) / num_positives
        
        # Only average over samples that have positive pairs
        valid_mask = label_matrix.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            return loss[valid_mask].mean()
        return torch.tensor(0.0, device=device, requires_grad=True)


class CombinedOCRLoss(nn.Module):
    """
    Combined loss function for OCR training.
    Combines CTC loss with optional contrastive loss.
    """
    
    def __init__(
        self,
        ctc_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        use_contrastive: bool = True,
        temperature: float = 0.07
    ):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        self.contrastive_loss = MultiFrameContrastiveLoss(temperature=temperature)
        
        self.ctc_weight = ctc_weight
        self.contrastive_weight = contrastive_weight
        self.use_contrastive = use_contrastive
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        frame_features: torch.Tensor = None
    ) -> tuple:
        """
        Compute combined loss.
        
        Args:
            log_probs: CTC log probabilities [T, B, C]
            targets: Target sequences (concatenated)
            input_lengths: Length of each input sequence
            target_lengths: Length of each target sequence
            frame_features: Optional frame features for contrastive loss [B, T, D]
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # CTC Loss
        ctc = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        loss_dict = {'ctc': ctc.item()}
        total_loss = self.ctc_weight * ctc
        
        # Contrastive Loss (if features provided)
        if self.use_contrastive and frame_features is not None:
            contrastive = self.contrastive_loss(frame_features)
            loss_dict['contrastive'] = contrastive.item()
            total_loss = total_loss + self.contrastive_weight * contrastive
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class FocalCTCLoss(nn.Module):
    """
    Focal CTC Loss - down-weights easy examples.
    Helps focus on harder samples during training.
    """
    
    def __init__(self, blank: int = 0, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True, reduction='none')
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal CTC loss."""
        # Get per-sample CTC loss
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # Convert to probability (lower loss = higher prob)
        # Approximate probability as exp(-loss)
        p = torch.exp(-ctc_loss)
        
        # Focal weight: (1 - p)^gamma
        focal_weight = (1 - p) ** self.gamma
        
        # Apply focal weight
        focal_loss = self.alpha * focal_weight * ctc_loss
        
        return focal_loss.mean()


class LabelSmoothingCTCLoss(nn.Module):
    """
    CTC Loss with label smoothing effect via temperature scaling.
    Prevents overconfident predictions.
    """
    
    def __init__(self, blank: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True, reduction='mean')
        self.smoothing = smoothing
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss with implicit label smoothing.
        
        We add uniform noise to log_probs before computing CTC,
        which has a similar effect to label smoothing.
        """
        if self.smoothing > 0 and self.training:
            num_classes = log_probs.size(-1)
            # Add uniform smoothing
            smooth_log_probs = (1 - self.smoothing) * log_probs + \
                              self.smoothing * torch.full_like(log_probs, -torch.log(torch.tensor(num_classes)))
        else:
            smooth_log_probs = log_probs
        
        return self.ctc_loss(smooth_log_probs, targets, input_lengths, target_lengths)


def get_loss_function(
    loss_type: str = 'combined',
    **kwargs
) -> nn.Module:
    """Factory function to create loss function.
    
    Args:
        loss_type: One of 'ctc', 'combined', 'focal', 'smoothing'
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function module
    """
    if loss_type == 'ctc':
        return nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
    elif loss_type == 'combined':
        return CombinedOCRLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalCTCLoss(**kwargs)
    elif loss_type == 'smoothing':
        return LabelSmoothingCTCLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
