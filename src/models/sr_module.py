"""Super-Resolution modules for enhancing low-resolution license plate images."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with two conv layers."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class ChannelAttention(nn.Module):
    """Channel attention module (SE-like)."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualChannelAttentionBlock(nn.Module):
    """Residual block with channel attention (RCAN-style)."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = ChannelAttention(channels, reduction)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        return out + residual


class LightweightSRModule(nn.Module):
    """
    Lightweight Super-Resolution module for license plate enhancement.
    Uses residual blocks with upscaling via PixelShuffle.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 4,
        upscale_factor: int = 2
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        # Initial feature extraction
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.PReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_features) for _ in range(num_blocks)
        ])
        
        # Global residual connection
        self.conv_mid = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # Upscaling
        if upscale_factor == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
        elif upscale_factor == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
        else:
            # Fallback to interpolation
            self.upscale = nn.Sequential(
                nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.PReLU()
            )
        
        # Output reconstruction
        self.conv_out = nn.Conv2d(num_features, in_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input LR images [B, C, H, W]
        Returns:
            SR images [B, C, H*scale, W*scale]
        """
        # Bilinear upscale for residual connection
        x_up = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        
        # Feature extraction
        feat = self.conv_in(x)
        
        # Residual learning
        res = self.res_blocks(feat)
        res = self.conv_mid(res)
        feat = feat + res  # Global residual
        
        # Upscale
        feat = self.upscale(feat)
        
        # Output + skip connection
        out = self.conv_out(feat) + x_up
        
        return out


class RCANLiteSR(nn.Module):
    """
    Lightweight RCAN-inspired SR module with channel attention.
    More powerful than basic ResNet SR but still efficient.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 6,
        reduction: int = 16,
        upscale_factor: int = 2
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        # Shallow feature extraction
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        
        # Residual channel attention blocks
        self.rcab_blocks = nn.Sequential(*[
            ResidualChannelAttentionBlock(num_features, reduction)
            for _ in range(num_blocks)
        ])
        
        # Feature refinement
        self.conv_mid = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # Upscaling module
        upscale_modules = []
        if upscale_factor >= 2:
            upscale_modules.extend([
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ])
        if upscale_factor >= 4:
            upscale_modules.extend([
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ])
        self.upscale = nn.Sequential(*upscale_modules) if upscale_modules else nn.Identity()
        
        # Output reconstruction
        self.conv_out = nn.Conv2d(num_features, in_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input LR images [B, C, H, W]
        Returns:
            SR images [B, C, H*scale, W*scale]
        """
        # Skip connection via bilinear upscale
        x_up = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        
        # Feature extraction
        feat = self.conv_in(x)
        shallow_feat = feat
        
        # Deep feature extraction with RCAB
        feat = self.rcab_blocks(feat)
        feat = self.conv_mid(feat)
        
        # Global residual
        feat = feat + shallow_feat
        
        # Upscale
        feat = self.upscale(feat)
        
        # Reconstruction with skip
        out = self.conv_out(feat) + x_up
        
        return out


class SRPipeline(nn.Module):
    """
    Complete SR + OCR pipeline for joint training.
    SR module enhances images before feeding to OCR backbone.
    """
    
    def __init__(
        self,
        sr_module: nn.Module,
        ocr_module: nn.Module,
        use_sr: bool = True,
        sr_weight: float = 0.1
    ):
        super().__init__()
        self.sr = sr_module
        self.ocr = ocr_module
        self.use_sr = use_sr
        self.sr_weight = sr_weight
    
    def forward(
        self,
        x: torch.Tensor,
        hr_target: torch.Tensor = None
    ):
        """
        Args:
            x: Input LR images [B, T, C, H, W] (multi-frame)
            hr_target: HR ground truth for SR loss (optional, training only)
        
        Returns:
            OCR output, SR loss (if hr_target provided)
        """
        B, T, C, H, W = x.shape
        sr_loss = None
        
        if self.use_sr:
            # Apply SR to each frame
            x_flat = x.view(B * T, C, H, W)
            x_sr = self.sr(x_flat)  # [B*T, C, H', W']
            
            # Compute SR loss if target available
            if hr_target is not None and self.training:
                hr_flat = hr_target.view(B * T, C, -1, -1)
                # Resize hr_target to match sr output if needed
                if x_sr.shape != hr_flat.shape:
                    hr_flat = F.interpolate(hr_flat, size=x_sr.shape[-2:], mode='bilinear', align_corners=False)
                sr_loss = F.l1_loss(x_sr, hr_flat) * self.sr_weight
            
            # Reshape for OCR
            _, _, H_new, W_new = x_sr.shape
            x = x_sr.view(B, T, C, H_new, W_new)
        
        # Run OCR
        ocr_output = self.ocr(x)
        
        if sr_loss is not None:
            return ocr_output, sr_loss
        return ocr_output
    
    def set_sr_enabled(self, enabled: bool):
        """Enable/disable SR module at inference time."""
        self.use_sr = enabled


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features for SR training."""
    
    def __init__(self, layers: list = None):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        
        # Default: conv3_3 (layer 15)
        if layers is None:
            layers = [15]
        
        self.layers = layers
        self.features = nn.ModuleList()
        
        prev_idx = 0
        for idx in layers:
            self.features.append(nn.Sequential(*list(vgg.children())[prev_idx:idx+1]))
            prev_idx = idx + 1
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between SR and HR images."""
        loss = 0.0
        
        for feature_extractor in self.features:
            sr_feat = feature_extractor(sr)
            hr_feat = feature_extractor(hr)
            loss += F.mse_loss(sr_feat, hr_feat)
        
        return loss


def get_sr_module(sr_type: str, **kwargs) -> nn.Module:
    """Factory function to create SR module.
    
    Args:
        sr_type: One of 'lightweight', 'rcan_lite'
        **kwargs: Additional arguments
    
    Returns:
        SR module
    """
    sr_modules = {
        'lightweight': LightweightSRModule,
        'rcan_lite': RCANLiteSR,
    }
    
    if sr_type not in sr_modules:
        raise ValueError(f"Unknown SR type: {sr_type}. Choose from {list(sr_modules.keys())}")
    
    return sr_modules[sr_type](**kwargs)
