"""
PaliGemma-3B LoRA Fine-tuning for License Plate OCR from Blurry Images.

This script combines:
1. Lightweight Deblur/Super-Resolution module (before VLM)
2. PaliGemma-3B Vision-Language Model with LoRA
3. Multi-frame fusion for license plate recognition

Usage:
    python lora/train.py --epochs 20 --batch_size 4 --lr 2e-4
"""

import os
import sys
import argparse
import json
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm
import cv2
import numpy as np

# Hugging Face imports
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for PaliGemma LoRA training."""
    
    # Model settings
    model_name: str = "google/paligemma-3b-pt-224"
    use_4bit: bool = True
    use_8bit: bool = False
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Deblur module settings
    use_deblur: bool = True
    deblur_type: str = "lightweight"  # "lightweight", "esrgan_lite"
    deblur_num_blocks: int = 4
    deblur_features: int = 32
    freeze_deblur: bool = False  # Set to True after pretraining
    
    # Data settings
    data_root: str = "data/train"
    test_root: str = "data/test"
    val_split_file: str = "data/val_tracks.json"
    img_size: int = 224
    max_length: int = 20
    data_limit: int = 0  # 0 = use all data, >0 = limit number of training samples
    
    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    deblur_lr: float = 1e-3  # Higher LR for deblur module
    epochs: int = 20
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Prompt template
    prompt_template: str = "OCR license plate text:"
    
    # Output settings
    output_dir: str = "results/paligemma_lora"
    save_steps: int = 500
    eval_steps: int = 200
    logging_steps: int = 50
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 4
    use_flash_attn: bool = False
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


# =============================================================================
# Lightweight Deblur Module
# =============================================================================

class ResidualBlock(nn.Module):
    """Basic residual block for deblur network."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LightweightDeblur(nn.Module):
    """
    Lightweight deblur/enhancement module.
    Runs BEFORE the VLM to clean up blurry license plate images.
    
    Architecture:
    - Shallow encoder-decoder with residual blocks
    - Channel attention for adaptive feature refinement
    - Skip connections for preserving details
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 32,
        num_blocks: int = 4,
    ):
        super().__init__()
        
        # Encoder
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks with channel attention
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(num_features),
                ChannelAttention(num_features)
            )
            for _ in range(num_blocks)
        ])
        
        # Decoder
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Blurry input images [B, C, H, W], normalized to [0, 1] or [-1, 1]
        Returns:
            Enhanced images [B, C, H, W], same range as input
        """
        # Feature extraction
        feat = self.conv_in(x)
        
        # Residual blocks
        for block in self.res_blocks:
            feat = block(feat)
        
        # Reconstruction with skip connection
        out = self.conv_out(feat) + x
        
        return out


class ESRGANLiteDeblur(nn.Module):
    """
    Real-ESRGAN inspired lightweight deblur module.
    Uses RRDB-lite blocks for better quality at reasonable compute.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 32,
        num_blocks: int = 4,
        growth_channels: int = 16,
    ):
        super().__init__()
        
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1)
        
        # Dense residual blocks (simplified RRDB)
        self.body = nn.Sequential(*[
            RRDBLite(num_features, growth_channels)
            for _ in range(num_blocks)
        ])
        
        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv_last = nn.Conv2d(num_features, in_channels, 3, padding=1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.lrelu(self.conv_first(x))
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat * 0.2  # Residual scaling
        out = self.conv_last(self.lrelu(feat))
        return out + x


class RRDBLite(nn.Module):
    """Simplified Residual-in-Residual Dense Block."""
    
    def __init__(self, nf: int = 32, gc: int = 16):
        super().__init__()
        self.dense1 = DenseBlockLite(nf, gc)
        self.dense2 = DenseBlockLite(nf, gc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dense1(x)
        out = self.dense2(out)
        return out * 0.2 + x


class DenseBlockLite(nn.Module):
    """Simplified dense block."""
    
    def __init__(self, nf: int = 32, gc: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding=1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        return x3 * 0.2 + x


def get_deblur_module(config: TrainingConfig) -> nn.Module:
    """Factory function to create deblur module."""
    if config.deblur_type == "esrgan_lite":
        return ESRGANLiteDeblur(
            num_features=config.deblur_features,
            num_blocks=config.deblur_num_blocks,
        )
    else:
        return LightweightDeblur(
            num_features=config.deblur_features,
            num_blocks=config.deblur_num_blocks,
        )


# =============================================================================
# Dataset
# =============================================================================

class LicensePlateDataset(Dataset):
    """Dataset for PaliGemma license plate OCR training."""
    
    def __init__(
        self,
        root_dir: str,
        processor: Any,
        mode: str = "train",
        val_split_file: str = None,
        split_ratio: float = 0.9,
        img_size: int = 224,
        prompt: str = "OCR license plate text:",
        max_length: int = 20,
        seed: int = 42,
        full_train: bool = False,
        data_limit: int = 0,
    ):
        self.processor = processor
        self.mode = mode
        self.img_size = img_size
        self.prompt = prompt
        self.max_length = max_length
        self.data_limit = data_limit
        self.samples = []
        
        # Find all track folders
        search_path = os.path.join(root_dir, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print(f"WARNING: No tracks found in {root_dir}")
            return
        
        # Load or create train/val split
        train_tracks, val_tracks = self._load_or_create_split(
            all_tracks, val_split_file, split_ratio, seed, full_train
        )
        
        selected_tracks = train_tracks if mode == "train" else val_tracks
        print(f"[{mode.upper()}] Loading {len(selected_tracks)} tracks...")
        
        # Index samples
        self._index_samples(selected_tracks)
        
        # Apply data_limit if set (truncate after shuffling for randomness)
        if self.data_limit > 0 and len(self.samples) > self.data_limit:
            random.seed(seed)
            random.shuffle(self.samples)
            self.samples = self.samples[:self.data_limit]
            print(f"[{mode.upper()}] Data limit applied: {self.data_limit} samples")
        
        print(f"[{mode.upper()}] Total samples: {len(self.samples)}")
    
    def _load_or_create_split(
        self,
        all_tracks: List[str],
        val_split_file: str,
        split_ratio: float,
        seed: int,
        full_train: bool,
    ) -> Tuple[List[str], List[str]]:
        """Load existing split or create new one."""
        if full_train:
            return all_tracks, []
        
        if val_split_file and os.path.exists(val_split_file):
            with open(val_split_file, 'r') as f:
                val_ids = set(json.load(f))
            
            train_tracks = [t for t in all_tracks if os.path.basename(t) not in val_ids]
            val_tracks = [t for t in all_tracks if os.path.basename(t) in val_ids]
            
            if val_tracks:
                return train_tracks, val_tracks
        
        # Create new split
        random.seed(seed)
        all_tracks_shuffled = all_tracks.copy()
        random.shuffle(all_tracks_shuffled)
        
        split_idx = int(len(all_tracks_shuffled) * split_ratio)
        train_tracks = all_tracks_shuffled[:split_idx]
        val_tracks = all_tracks_shuffled[split_idx:]
        
        return train_tracks, val_tracks
    
    def _index_samples(self, tracks: List[str]):
        """Index samples from tracks."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue
                
                # Get all LR images
                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                
                if lr_files:
                    self.samples.append({
                        'paths': lr_files,
                        'label': label.upper(),
                        'track_id': os.path.basename(track_path),
                    })
            except Exception as e:
                continue
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load sample with multi-frame fusion (average frames)."""
        item = self.samples[idx]
        paths = item['paths']
        label = item['label']
        track_id = item['track_id']
        
        # Load and preprocess images
        images = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            images.append(img)
        
        # Average frames for multi-frame fusion
        avg_image = np.mean(images, axis=0).astype(np.uint8)
        
        # Also keep original frames for deblur module
        frames_tensor = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for img in images
        ])
        
        # PaliGemma format: prepend <image> token for the image
        # Full text for training: <image> + prompt + answer
        full_text = f"<image>{self.prompt} {label}"
        
        # Process image and full text together
        inputs = self.processor(
            images=avg_image,
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length + 256,  # Extra space for image tokens
            truncation=True,
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Create labels for causal LM training
        # Labels = input_ids, but we mask the prompt part with -100
        labels = inputs['input_ids'].clone()
        
        # Get the prompt-only length to mask
        prompt_text = f"<image>{self.prompt}"
        prompt_inputs = self.processor(
            images=avg_image,
            text=prompt_text,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs['input_ids'].shape[1]
        
        # Mask prompt tokens (including image tokens) with -100
        labels[:prompt_len] = -100
        
        # Also mask padding tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        
        inputs['labels'] = labels
        inputs['frames'] = frames_tensor
        inputs['label_text'] = label
        inputs['track_id'] = track_id
        
        return inputs


class TestDataset(Dataset):
    """Dataset for test inference (no labels)."""
    
    def __init__(
        self,
        root_dir: str,
        processor: Any,
        img_size: int = 224,
        prompt: str = "OCR license plate text:",
    ):
        self.processor = processor
        self.img_size = img_size
        self.prompt = prompt
        self.samples = []
        
        # Find all track folders
        search_path = os.path.join(root_dir, "track_*")
        all_tracks = sorted(glob.glob(search_path))
        
        print(f"[TEST] Loading {len(all_tracks)} tracks...")
        
        for track_path in all_tracks:
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png")) +
                glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            if lr_files:
                self.samples.append({
                    'paths': lr_files,
                    'track_id': os.path.basename(track_path),
                })
        
        print(f"[TEST] Total samples: {len(self.samples)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        paths = item['paths']
        track_id = item['track_id']
        
        images = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            images.append(img)
        
        avg_image = np.mean(images, axis=0).astype(np.uint8)
        
        frames_tensor = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for img in images
        ])
        
        # PaliGemma format: prepend <image> token
        prompt_text = f"<image>{self.prompt}"
        inputs = self.processor(
            images=avg_image,
            text=prompt_text,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['frames'] = frames_tensor
        inputs['track_id'] = track_id
        
        return inputs


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function."""
    result = {}
    
    # Stack tensors
    for key in batch[0].keys():
        if key in ['label_text', 'track_id']:
            result[key] = [item[key] for item in batch]
        elif key == 'frames':
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = torch.stack([item[key] for item in batch])
    
    return result


# =============================================================================
# Model with Deblur
# =============================================================================

class PaliGemmaWithDeblur(nn.Module):
    """
    Combined model: Deblur Module + PaliGemma with LoRA.
    
    The deblur module enhances blurry images before feeding to VLM.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        processor: Any,
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        
        # Initialize deblur module
        if config.use_deblur:
            self.deblur = get_deblur_module(config)
            if config.freeze_deblur:
                for param in self.deblur.parameters():
                    param.requires_grad = False
        else:
            self.deblur = None
        
        # Initialize PaliGemma with quantization
        bnb_config = None
        if config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif config.use_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa" if config.use_flash_attn else "eager",
        )

        self.vlm.config.use_cache = False  # Disable caching for training
        
        # Prepare for k-bit training
        if bnb_config:
            self.vlm = prepare_model_for_kbit_training(
                self.vlm, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.vlm = get_peft_model(self.vlm, lora_config)
        self.vlm.print_trainable_parameters()
    
    def enhance_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Apply deblur enhancement to images."""
        if self.deblur is None:
            return pixel_values
        
        # pixel_values: [B, C, H, W], normalized
        # Deblur expects [0, 1] or [-1, 1] range
        # Cast to deblur module dtype (float32) then cast back
        input_dtype = pixel_values.dtype
        enhanced = self.deblur(pixel_values.float())
        
        return enhanced.to(input_dtype)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with deblur + VLM."""
        # Enhance images with deblur
        enhanced_pixels = self.enhance_images(pixel_values)
        
        # Disable external autocast for VLM — the quantized model handles its
        # own mixed-precision via bnb_4bit_compute_dtype.  External autocast
        # causes a dtype mismatch (bfloat16 attn_mask vs float32 query from
        # kbit upcasting) inside SDPA attention.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            outputs = self.vlm(
                pixel_values=enhanced_pixels,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 20,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text from enhanced images."""
        enhanced_pixels = self.enhance_images(pixel_values)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            outputs = self.vlm.generate(
                pixel_values=enhanced_pixels,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        return outputs


# =============================================================================
# Training Functions
# =============================================================================

def compute_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Compute exact match accuracy."""
    if not predictions or not targets:
        return 0.0
    
    correct = sum(p.strip().upper() == t.strip().upper() for p, t in zip(predictions, targets))
    return correct / len(predictions)


def decode_predictions(
    generated_ids: torch.Tensor,
    processor: Any,
    prompt_length: int = 0,
) -> List[str]:
    """Decode generated token IDs to text."""
    predictions = []
    for ids in generated_ids:
        # Skip prompt tokens
        if prompt_length > 0:
            ids = ids[prompt_length:]
        
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        # Clean up - extract only alphanumeric
        cleaned = ''.join(c for c in text if c.isalnum())
        predictions.append(cleaned.upper())
    
    return predictions


def decode_predictions_with_confidence(
    model: 'PaliGemmaWithDeblur',
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    processor: Any,
    max_new_tokens: int = 20,
) -> List[Tuple[str, float]]:
    """Generate predictions with per-sample confidence scores.
    
    Confidence is computed as the mean probability of generated tokens
    (excluding special/padding tokens).
    
    Returns:
        List of (prediction_text, confidence) tuples.
    """
    # Enhance images
    enhanced_pixels = model.enhance_images(pixel_values)
    prompt_length = input_ids.shape[1]
    
    # Generate with output scores
    outputs = model.vlm.generate(
        pixel_values=enhanced_pixels,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    
    generated_ids = outputs.sequences
    scores = outputs.scores  # tuple of (vocab_size,) tensors, one per generated step
    
    results = []
    batch_size = generated_ids.shape[0]
    
    for b in range(batch_size):
        # Decode text
        ids = generated_ids[b][prompt_length:]
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        cleaned = ''.join(c for c in text if c.isalnum()).upper()
        
        # Compute confidence from generation scores
        if scores:
            token_confidences = []
            eos_id = processor.tokenizer.eos_token_id
            pad_id = processor.tokenizer.pad_token_id
            for step_idx, score in enumerate(scores):
                probs = torch.softmax(score[b], dim=-1)
                token_id = generated_ids[b][prompt_length + step_idx]
                if token_id == eos_id or token_id == pad_id:
                    break
                token_confidences.append(probs[token_id].item())
            
            confidence = float(np.mean(token_confidences)) if token_confidences else 0.0
        else:
            confidence = 0.0
        
        results.append((cleaned, confidence))
    
    return results


def train_epoch(
    model: PaliGemmaWithDeblur,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: TrainingConfig,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        # Move to device
        pixel_values = batch['pixel_values'].to(config.device, dtype=torch.bfloat16)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device, dtype=torch.bool)
        labels = batch['labels'].to(config.device)
        
        # Forward pass (no outer autocast — the model disables it internally
        # to avoid dtype mismatch with gradient-checkpointed layers)
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{total_loss / num_batches:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}",
        })
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: PaliGemmaWithDeblur,
    val_loader: DataLoader,
    processor: Any,
    config: TrainingConfig,
) -> Dict[str, float]:
    """Evaluate model on validation set.
    
    Returns:
        Dict with keys: 'loss', 'recognition_rate', 'confidence_gap',
                        'avg_confidence', 'correct_confidence', 'incorrect_confidence'
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []    # list of (text, confidence)
    all_targets = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        pixel_values = batch['pixel_values'].to(config.device, dtype=torch.bfloat16)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device, dtype=torch.bool)
        labels = batch['labels'].to(config.device)
        targets = batch['label_text']
        
        # Compute loss
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        
        # Generate predictions with confidence
        preds_with_conf = decode_predictions_with_confidence(
            model, pixel_values, input_ids, attention_mask,
            processor, max_new_tokens=20,
        )
        all_predictions.extend(preds_with_conf)
        all_targets.extend(targets)
    
    avg_loss = total_loss / len(val_loader)
    
    # Compute Recognition Rate (exact match accuracy) and Confidence Gap
    correct_confidences = []
    incorrect_confidences = []
    total_correct = 0
    
    for (pred_text, conf), target in zip(all_predictions, all_targets):
        if pred_text.strip().upper() == target.strip().upper():
            total_correct += 1
            correct_confidences.append(conf)
        else:
            incorrect_confidences.append(conf)
    
    n_samples = len(all_predictions)
    recognition_rate = total_correct / n_samples if n_samples > 0 else 0.0
    
    avg_correct_conf = float(np.mean(correct_confidences)) if correct_confidences else 0.0
    avg_incorrect_conf = float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0
    avg_confidence = float(np.mean([c for _, c in all_predictions])) if all_predictions else 0.0
    
    # Confidence Gap = avg confidence of correct - avg confidence of incorrect
    # Higher gap means better calibrated (confident when right, uncertain when wrong)
    confidence_gap = avg_correct_conf - avg_incorrect_conf
    
    return {
        'loss': avg_loss,
        'recognition_rate': recognition_rate,
        'confidence_gap': confidence_gap,
        'avg_confidence': avg_confidence,
        'correct_confidence': avg_correct_conf,
        'incorrect_confidence': avg_incorrect_conf,
        'num_correct': total_correct,
        'num_total': n_samples,
    }


def save_checkpoint(
    model: PaliGemmaWithDeblur,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    config: TrainingConfig,
    filename: str = "checkpoint.pth",
):
    """Save training checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save deblur module separately
    if model.deblur is not None:
        deblur_path = os.path.join(config.output_dir, "deblur_module.pth")
        torch.save(model.deblur.state_dict(), deblur_path)
    
    # Save LoRA weights
    lora_path = os.path.join(config.output_dir, "paligemma_lora")
    model.vlm.save_pretrained(lora_path)
    
    # Save training state
    state = {
        'epoch': epoch,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    torch.save(state, os.path.join(config.output_dir, filename))
    
    print(f"Checkpoint saved to {config.output_dir}")


# =============================================================================
# Main Training Loop
# =============================================================================

def main(args):
    """Main training function."""
    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        use_4bit=not args.no_4bit,
        use_8bit=args.use_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_deblur=not args.no_deblur,
        deblur_type=args.deblur_type,
        use_flash_attn=args.flash_attn,
        data_root=args.data_root,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        epochs=args.epochs,
        output_dir=args.output_dir,
        seed=args.seed,
        data_limit=args.data_limit,
    )
    
    # Set seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    print("=" * 60)
    print("PaliGemma LoRA Training for License Plate OCR")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Deblur: {config.use_deblur} ({config.deblur_type})")
    print(f"LoRA r: {config.lora_r}, alpha: {config.lora_alpha}")
    print(f"Batch size: {config.batch_size} (accum: {config.gradient_accumulation_steps})")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Data limit: {'all' if config.data_limit == 0 else config.data_limit}")
    print(f"Epochs: {config.epochs}")
    print("=" * 60)
    
    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(config.model_name, use_fast=True)
    
    # Ensure padding token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = LicensePlateDataset(
        root_dir=config.data_root,
        processor=processor,
        mode="train",
        val_split_file=config.val_split_file,
        img_size=config.img_size,
        prompt=config.prompt_template,
        seed=config.seed,
        data_limit=config.data_limit,
    )
    
    val_dataset = LicensePlateDataset(
        root_dir=config.data_root,
        processor=processor,
        mode="val",
        val_split_file=config.val_split_file,
        img_size=config.img_size,
        prompt=config.prompt_template,
        seed=config.seed,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Create model
    print("\nLoading model...")
    model = PaliGemmaWithDeblur(config, processor)
    
    # Move deblur to device (VLM already on device via device_map)
    if model.deblur is not None:
        model.deblur = model.deblur.to(config.device)
    
    # Create optimizer with different LR for deblur
    param_groups = []
    
    # VLM parameters (LoRA)
    vlm_params = [p for p in model.vlm.parameters() if p.requires_grad]
    param_groups.append({
        'params': vlm_params,
        'lr': config.learning_rate,
        'weight_decay': config.weight_decay,
    })
    
    # Deblur parameters (if trainable)
    if model.deblur is not None and not config.freeze_deblur:
        deblur_params = list(model.deblur.parameters())
        param_groups.append({
            'params': deblur_params,
            'lr': config.deblur_lr,
            'weight_decay': config.weight_decay,
        })
    
    optimizer = torch.optim.AdamW(param_groups)
    
    # Create scheduler
    num_training_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, config, epoch
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if len(val_dataset) > 0:
            metrics = evaluate(model, val_loader, processor, config)
            val_loss = metrics['loss']
            val_rr = metrics['recognition_rate']
            conf_gap = metrics['confidence_gap']
            
            print(f"\n{'='*50}")
            print(f"  Val Loss:              {val_loss:.4f}")
            print(f"  Recognition Rate:      {val_rr:.4f} ({metrics['num_correct']}/{metrics['num_total']})")
            print(f"  Confidence Gap:        {conf_gap:+.4f}")
            print(f"  Avg Confidence:        {metrics['avg_confidence']:.4f}")
            print(f"  Correct Confidence:    {metrics['correct_confidence']:.4f}")
            print(f"  Incorrect Confidence:  {metrics['incorrect_confidence']:.4f}")
            print(f"{'='*50}")
            
            # Save best model based on recognition rate
            if val_rr > best_acc:
                best_acc = val_rr
                save_checkpoint(model, optimizer, epoch, best_acc, config, "best.pth")
                print(f">>> New best Recognition Rate: {best_acc:.4f} <<<")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, best_acc, config, f"epoch_{epoch+1}.pth")
    
    # Save final model
    save_checkpoint(model, optimizer, config.epochs - 1, best_acc, config, "final.pth")
    print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def inference(args):
    """Run inference on test set."""
    config = TrainingConfig(
        model_name=args.model_name,
        use_4bit=not args.no_4bit,
        use_deblur=not args.no_deblur,
        deblur_type=args.deblur_type,
        use_flash_attn=args.flash_attn,
        test_root=args.test_root,
        output_dir=args.output_dir,
    )
    
    print("Loading model for inference...")
    processor = AutoProcessor.from_pretrained(config.model_name, use_fast=True)
    
    # Load model
    model = PaliGemmaWithDeblur(config, processor)
    
    # Load checkpoint
    checkpoint_dir = config.output_dir
    
    # Load deblur weights
    deblur_path = os.path.join(checkpoint_dir, "deblur_module.pth")
    if model.deblur is not None and os.path.exists(deblur_path):
        model.deblur.load_state_dict(torch.load(deblur_path, weights_only=True))
        model.deblur = model.deblur.to(config.device)
    
    # Load LoRA weights
    lora_path = os.path.join(checkpoint_dir, "paligemma_lora")
    if os.path.exists(lora_path):
        from peft import PeftModel
        model.vlm = PeftModel.from_pretrained(model.vlm.base_model, lora_path)
    
    model.eval()
    
    # Create test dataset
    test_dataset = TestDataset(
        root_dir=config.test_root,
        processor=processor,
        img_size=config.img_size,
        prompt=config.prompt_template,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    
    # Run inference
    results = []
    
    for batch in tqdm(test_loader, desc="Inference"):
        pixel_values = batch['pixel_values'].to(config.device, dtype=torch.bfloat16)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device, dtype=torch.bool)
        track_id = batch['track_id'][0]
        
        generated = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
        )
        
        prediction = decode_predictions(generated, processor, input_ids.shape[1])[0]
        results.append(f"{track_id}\t{prediction}")
    
    # Save results
    output_file = os.path.join(config.output_dir, "prediction.txt")
    with open(output_file, 'w') as f:
        f.write('\n'.join(results))
    
    print(f"Predictions saved to {output_file}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaliGemma LoRA Training")
    
    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"])
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="google/paligemma-3b-pt-224")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    
    # LoRA settings
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Deblur settings
    parser.add_argument("--no_deblur", action="store_true", help="Disable deblur module")
    parser.add_argument("--deblur_type", type=str, default="lightweight",
                       choices=["lightweight", "esrgan_lite"])
    parser.add_argument("--flash_attn", action="store_true", help="Use SDPA (PyTorch native scaled dot product attention)")
    
    # Data settings
    parser.add_argument("--data_root", type=str, default="data/train")
    parser.add_argument("--test_root", type=str, default="data/test")
    parser.add_argument("--data_limit", type=int, default=0,
                       help="Limit number of training samples (0 = use all data)")
    
    # Training settings
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/paligemma_lora")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main(args)
    else:
        inference(args)
