"""ResTranOCR: ResNet34 + Transformer architecture (Advanced) with STN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import ResNetFeatureExtractor, AttentionFusion, PositionalEncoding, STNBlock
from src.models.temporal_fusion import get_temporal_fusion, QualityAwareAttentionFusion, HybridTemporalFusion

class ResTranOCR(nn.Module):
    """
    Modern OCR architecture using optional STN, ResNet34 and Transformer.
    Pipeline: Input (5 frames) -> [Optional STN] -> ResNet34 -> Temporal Fusion -> Transformer -> CTC Head
    
    Supports multiple temporal fusion strategies:
    - attention: Original simple attention fusion
    - quality: Quality-aware attention with per-frame quality scoring
    - transformer: Cross-frame transformer for global temporal modeling
    - hybrid: Combines quality-aware and transformer approaches
    """
    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        temporal_fusion_type: str = "attention"
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.temporal_fusion_type = temporal_fusion_type
        
        # 1. Spatial Transformer Network
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone: ResNet34
        self.backbone = ResNetFeatureExtractor(pretrained=True)
        
        # 3. Temporal Fusion (supports multiple strategies)
        if temporal_fusion_type == "attention":
            self.fusion = AttentionFusion(channels=self.cnn_channels)
        elif temporal_fusion_type in ["quality", "transformer", "hybrid"]:
            self.fusion = get_temporal_fusion(
                fusion_type=temporal_fusion_type,
                channels=self.cnn_channels,
                num_frames=5
            )
        else:
            raise ValueError(f"Unknown fusion type: {temporal_fusion_type}")
        
        # 4. Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 5. Prediction Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
        Returns:
            Logits: [Batch, Seq_Len, Num_Classes]
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)  # [B*F, C, H, W]
        
        if self.use_stn:
            theta = self.stn(x_flat)  # [B*F, 2, 3]
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat
        
        features = self.backbone(x_aligned)  # [B*F, 512, 1, W']
        fused = self.fusion(features)       # [B, 512, 1, W']
        
        # Prepare for Transformer: [B, C, 1, W'] -> [B, W', C]
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        # Add Positional Encoding and pass through Transformer
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) # [B, W', C]
        
        out = self.head(seq_out)              # [B, W', Num_Classes]
        return out.log_softmax(2)