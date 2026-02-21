"""Test-Time Augmentation (TTA) for improved inference accuracy."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable
import numpy as np


class TestTimeAugmentation:
    """
    Apply multiple augmentations at test time and aggregate predictions.
    Improves robustness and accuracy at the cost of inference speed.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        augmentations: List[str] = None,
        aggregation: str = 'mean'
    ):
        """
        Args:
            model: OCR model
            device: Device to run inference on
            augmentations: List of augmentation names to apply
                          Options: 'original', 'hflip', 'brightness', 'contrast',
                                   'scale_up', 'scale_down', 'rotate_cw', 'rotate_ccw'
            aggregation: How to aggregate predictions ('mean', 'max', 'vote')
        """
        self.model = model
        self.device = device
        self.aggregation = aggregation
        
        # Default augmentations (conservative for license plates)
        if augmentations is None:
            augmentations = ['original', 'brightness_up', 'brightness_down', 'contrast']
        
        self.augmentations = augmentations
        self._build_transforms()
    
    def _build_transforms(self):
        """Build augmentation functions."""
        self.transforms = {
            'original': lambda x: x,
            'hflip': lambda x: torch.flip(x, dims=[-1]),  # Horizontal flip
            'brightness_up': lambda x: torch.clamp(x + 0.1, -1, 1),
            'brightness_down': lambda x: torch.clamp(x - 0.1, -1, 1),
            'contrast': lambda x: x * 1.1,
            'scale_up': self._scale_up,
            'scale_down': self._scale_down,
            'rotate_cw': lambda x: self._rotate(x, 2),   # 2 degrees clockwise
            'rotate_ccw': lambda x: self._rotate(x, -2),  # 2 degrees counter-clockwise
        }
    
    def _scale_up(self, x: torch.Tensor, factor: float = 1.05) -> torch.Tensor:
        """Scale up image slightly."""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        new_h, new_w = int(H * factor), int(W * factor)
        scaled = F.interpolate(x_flat, size=(new_h, new_w), mode='bilinear', align_corners=False)
        # Center crop back to original size
        start_h, start_w = (new_h - H) // 2, (new_w - W) // 2
        cropped = scaled[:, :, start_h:start_h+H, start_w:start_w+W]
        return cropped.view(B, T, C, H, W)
    
    def _scale_down(self, x: torch.Tensor, factor: float = 0.95) -> torch.Tensor:
        """Scale down image slightly and pad."""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        new_h, new_w = int(H * factor), int(W * factor)
        scaled = F.interpolate(x_flat, size=(new_h, new_w), mode='bilinear', align_corners=False)
        # Pad back to original size
        pad_h, pad_w = (H - new_h) // 2, (W - new_w) // 2
        padded = F.pad(scaled, (pad_w, W - new_w - pad_w, pad_h, H - new_h - pad_h), value=-1)
        return padded.view(B, T, C, H, W)
    
    def _rotate(self, x: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Rotate image by small angle."""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        # Create rotation matrix
        angle_rad = torch.tensor(angle_deg * np.pi / 180, device=x.device)
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
        
        # Affine transformation matrix
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], device=x.device, dtype=x.dtype).unsqueeze(0).expand(B * T, -1, -1)
        
        grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
        rotated = F.grid_sample(x_flat, grid, align_corners=False, padding_mode='border')
        
        return rotated.view(B, T, C, H, W)
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run TTA inference.
        
        Args:
            images: Input images [B, T, C, H, W]
        
        Returns:
            Aggregated log probabilities [B, Seq, Classes]
        """
        self.model.eval()
        all_log_probs = []
        
        for aug_name in self.augmentations:
            if aug_name not in self.transforms:
                continue
            
            transform = self.transforms[aug_name]
            augmented = transform(images.clone())
            
            log_probs = self.model(augmented.to(self.device))
            all_log_probs.append(log_probs)
        
        # Aggregate predictions
        if self.aggregation == 'mean':
            # Average log probabilities
            aggregated = torch.stack(all_log_probs).mean(dim=0)
        elif self.aggregation == 'max':
            # Max log probabilities
            aggregated = torch.stack(all_log_probs).max(dim=0)[0]
        else:
            # Default to mean
            aggregated = torch.stack(all_log_probs).mean(dim=0)
        
        return aggregated
    
    def predict_with_confidence(
        self,
        images: torch.Tensor,
        idx2char: dict
    ) -> List[Tuple[str, float, float]]:
        """
        Run TTA inference with confidence and consistency scores.
        
        Args:
            images: Input images [B, T, C, H, W]
            idx2char: Index to character mapping
        
        Returns:
            List of (predicted_text, confidence, consistency_score) tuples
        """
        self.model.eval()
        all_log_probs = []
        all_predictions = []
        
        with torch.no_grad():
            for aug_name in self.augmentations:
                if aug_name not in self.transforms:
                    continue
                
                transform = self.transforms[aug_name]
                augmented = transform(images.clone())
                
                log_probs = self.model(augmented.to(self.device))
                all_log_probs.append(log_probs)
                
                # Decode this augmentation's prediction
                preds = self._greedy_decode(log_probs, idx2char)
                all_predictions.append(preds)
        
        # Aggregate log probs
        aggregated = torch.stack(all_log_probs).mean(dim=0)
        
        # Final predictions from aggregated
        final_preds = self._greedy_decode_with_conf(aggregated, idx2char)
        
        # Compute consistency score (how many augmentations agree)
        results = []
        B = images.size(0)
        for i in range(B):
            final_text, final_conf = final_preds[i]
            
            # Count how many augmentations predicted the same text
            aug_preds = [all_predictions[a][i] for a in range(len(all_predictions))]
            consistency = sum(1 for p in aug_preds if p == final_text) / len(aug_preds)
            
            results.append((final_text, final_conf, consistency))
        
        return results
    
    def _greedy_decode(self, log_probs: torch.Tensor, idx2char: dict) -> List[str]:
        """Simple greedy decoding."""
        predictions = log_probs.argmax(dim=-1)  # [B, T]
        results = []
        
        for pred in predictions:
            chars = []
            prev = -1
            for idx in pred.tolist():
                if idx != 0 and idx != prev:  # Not blank and not repeat
                    if idx in idx2char:
                        chars.append(idx2char[idx])
                prev = idx
            results.append(''.join(chars))
        
        return results
    
    def _greedy_decode_with_conf(
        self,
        log_probs: torch.Tensor,
        idx2char: dict
    ) -> List[Tuple[str, float]]:
        """Greedy decoding with confidence score."""
        probs = log_probs.exp()
        predictions = log_probs.argmax(dim=-1)  # [B, T]
        max_probs = probs.max(dim=-1)[0]  # [B, T]
        
        results = []
        
        for pred, mp in zip(predictions, max_probs):
            chars = []
            char_probs = []
            prev = -1
            
            for idx, prob in zip(pred.tolist(), mp.tolist()):
                if idx != 0 and idx != prev:
                    if idx in idx2char:
                        chars.append(idx2char[idx])
                        char_probs.append(prob)
                prev = idx
            
            text = ''.join(chars)
            conf = np.mean(char_probs) if char_probs else 0.0
            results.append((text, conf))
        
        return results


class MultiScaleTTA(TestTimeAugmentation):
    """
    Multi-scale TTA - resizes input to multiple scales and aggregates.
    Useful when license plates may appear at different sizes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        scales: List[float] = None,
        aggregation: str = 'mean'
    ):
        super().__init__(model, device, augmentations=['original'], aggregation=aggregation)
        self.scales = scales or [0.9, 1.0, 1.1]
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Run multi-scale TTA."""
        self.model.eval()
        B, T, C, H, W = images.shape
        all_log_probs = []
        
        for scale in self.scales:
            if scale == 1.0:
                scaled = images
            else:
                new_h, new_w = int(H * scale), int(W * scale)
                images_flat = images.view(B * T, C, H, W)
                scaled_flat = F.interpolate(images_flat, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                if scale > 1.0:
                    # Center crop
                    start_h, start_w = (new_h - H) // 2, (new_w - W) // 2
                    scaled_flat = scaled_flat[:, :, start_h:start_h+H, start_w:start_w+W]
                else:
                    # Pad
                    pad_h, pad_w = (H - new_h) // 2, (W - new_w) // 2
                    scaled_flat = F.pad(scaled_flat, (pad_w, W - new_w - pad_w, pad_h, H - new_h - pad_h), value=-1)
                
                scaled = scaled_flat.view(B, T, C, H, W)
            
            log_probs = self.model(scaled.to(self.device))
            all_log_probs.append(log_probs)
        
        return torch.stack(all_log_probs).mean(dim=0)


def create_tta(
    model: nn.Module,
    device: str = 'cuda',
    tta_type: str = 'standard',
    **kwargs
) -> TestTimeAugmentation:
    """Factory function to create TTA module.
    
    Args:
        model: OCR model
        device: Device for inference
        tta_type: 'standard' or 'multiscale'
        **kwargs: Additional arguments
    
    Returns:
        TTA module
    """
    if tta_type == 'standard':
        return TestTimeAugmentation(model, device, **kwargs)
    elif tta_type == 'multiscale':
        return MultiScaleTTA(model, device, **kwargs)
    else:
        raise ValueError(f"Unknown TTA type: {tta_type}")
