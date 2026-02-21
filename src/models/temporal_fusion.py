"""Advanced Temporal Attention Fusion modules for multi-frame feature aggregation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityAwareAttentionFusion(nn.Module):
    """
    Quality-aware attention fusion that learns to weight frames based on their quality.
    Uses a lightweight network to estimate per-frame quality scores.
    """
    
    def __init__(self, channels: int, num_frames: int = 5):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels
        
        # Quality assessment network - predicts quality score per frame
        self.quality_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels // 4, 1)
        )
        
        # Spatial attention for local quality assessment
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
        Returns:
            Fused feature map. Shape: [Batch, C, H, W]
        """
        total_frames, c, h, w = x.size()
        batch_size = total_frames // self.num_frames
        
        # Reshape to [Batch, Frames, C, H, W]
        x_view = x.view(batch_size, self.num_frames, c, h, w)
        
        # Compute frame-level quality scores: [Batch, Frames]
        quality_scores = []
        for f in range(self.num_frames):
            frame_feat = x_view[:, f]  # [B, C, H, W]
            score = self.quality_net(frame_feat)  # [B, 1]
            quality_scores.append(score)
        
        quality_scores = torch.cat(quality_scores, dim=1)  # [B, Frames]
        frame_weights = F.softmax(quality_scores, dim=1)   # [B, Frames]
        
        # Compute spatial attention for each frame
        spatial_weights = self.spatial_attn(x)  # [B*F, 1, H, W]
        spatial_weights = spatial_weights.view(batch_size, self.num_frames, 1, h, w)
        
        # Combine frame weights with spatial attention
        combined_weights = frame_weights.view(batch_size, self.num_frames, 1, 1, 1) * spatial_weights
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted fusion
        fused = (x_view * combined_weights).sum(dim=1)  # [B, C, H, W]
        
        return fused


class CrossFrameTransformerFusion(nn.Module):
    """
    Transformer-based cross-frame feature aggregation.
    Uses self-attention across frames to capture temporal dependencies.
    """
    
    def __init__(
        self,
        channels: int,
        num_frames: int = 5,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels
        
        # Reduce spatial dims before transformer
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, None))  # Keep width for sequence
        
        # Learnable frame position embeddings
        self.frame_pos_embed = nn.Parameter(torch.randn(1, num_frames, channels) * 0.02)
        
        # Transformer encoder for cross-frame attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Aggregation: learnable query for fusion
        self.agg_query = nn.Parameter(torch.randn(1, 1, channels) * 0.02)
        self.agg_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection to restore spatial dims
        self.output_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
        Returns:
            Fused feature map. Shape: [Batch, C, 1, W]
        """
        total_frames, c, h, w = x.size()
        batch_size = total_frames // self.num_frames
        
        # Pool height: [B*F, C, 1, W]
        x_pooled = self.spatial_pool(x)
        
        # Reshape: [B, F, C, W]
        x_view = x_pooled.squeeze(2).view(batch_size, self.num_frames, c, w)
        
        # Average over width to get frame-level features: [B, F, C]
        frame_features = x_view.mean(dim=-1)
        
        # Add frame position embeddings
        frame_features = frame_features + self.frame_pos_embed
        
        # Cross-frame transformer encoding
        encoded = self.transformer(frame_features)  # [B, F, C]
        
        # Aggregate using attention with learnable query
        agg_query = self.agg_query.expand(batch_size, -1, -1)  # [B, 1, C]
        fused, _ = self.agg_attn(agg_query, encoded, encoded)  # [B, 1, C]
        fused = self.output_proj(fused.squeeze(1))  # [B, C]
        
        # Reshape to match expected output format
        # Use original spatial dims from input
        x_orig = x.view(batch_size, self.num_frames, c, h, w)
        
        # Weight original features by attention
        frame_weights = F.softmax(
            torch.einsum('bc,bfc->bf', fused, frame_features), dim=-1
        )  # [B, F]
        
        # Weighted sum of original features
        weighted = torch.einsum('bf,bfchw->bchw', frame_weights, x_orig)
        
        return weighted


class HybridTemporalFusion(nn.Module):
    """
    Hybrid fusion combining quality-aware attention and cross-frame transformer.
    Best of both worlds: local quality assessment + global temporal modeling.
    """
    
    def __init__(
        self,
        channels: int,
        num_frames: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_frames = num_frames
        
        # Quality-aware attention branch
        self.quality_branch = QualityAwareAttentionFusion(channels, num_frames)
        
        # Cross-frame transformer branch (lightweight)
        self.frame_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels)
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal branch: project temporal context back to spatial features
        self.temporal_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
        Returns:
            Fused feature map. Shape: [Batch, C, H, W]
        """
        total_frames, c, h, w = x.size()
        batch_size = total_frames // self.num_frames
        
        # Branch 1: Quality-aware attention
        quality_fused = self.quality_branch(x)  # [B, C, H, W]
        
        # Branch 2: Cross-frame embeddings
        x_view = x.view(batch_size, self.num_frames, c, h, w)
        frame_embeds = []
        for f in range(self.num_frames):
            embed = self.frame_embed(x_view[:, f])
            frame_embeds.append(embed)
        frame_embeds = torch.stack(frame_embeds, dim=1)  # [B, F, C]
        
        # Cross-frame attention to get temporal context
        temporal_ctx, _ = self.cross_attn(
            frame_embeds, frame_embeds, frame_embeds
        )  # [B, F, C]
        temporal_ctx = temporal_ctx.mean(dim=1)  # [B, C]
        
        # Project temporal context to spatial feature and broadcast
        temporal_spatial = self.temporal_proj(temporal_ctx)  # [B, C]
        temporal_spatial = temporal_spatial.view(batch_size, c, 1, 1).expand_as(quality_fused)  # [B, C, H, W]
        
        # Gate to combine quality and temporal branches
        quality_global = F.adaptive_avg_pool2d(quality_fused, 1).flatten(1)  # [B, C]
        gate_input = torch.cat([quality_global, temporal_ctx], dim=1)  # [B, 2C]
        gate_weight = self.gate(gate_input)  # [B, C]
        
        # Apply gate spatially
        gate_weight = gate_weight.view(batch_size, c, 1, 1)
        
        # Gated fusion: blend quality branch and temporal branch
        output = gate_weight * quality_fused + (1 - gate_weight) * temporal_spatial
        
        return output


def get_temporal_fusion(fusion_type: str, channels: int, num_frames: int = 5, **kwargs):
    """Factory function to create temporal fusion module.
    
    Args:
        fusion_type: One of 'attention', 'quality', 'transformer', 'hybrid'
        channels: Number of input channels
        num_frames: Number of frames to fuse
        **kwargs: Additional arguments passed to the fusion module
    
    Returns:
        Temporal fusion module
    """
    fusion_modules = {
        'quality': QualityAwareAttentionFusion,
        'transformer': CrossFrameTransformerFusion,
        'hybrid': HybridTemporalFusion,
    }
    
    if fusion_type not in fusion_modules:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Choose from {list(fusion_modules.keys())}")
    
    return fusion_modules[fusion_type](channels=channels, num_frames=num_frames, **kwargs)
