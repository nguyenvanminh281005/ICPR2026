#!/usr/bin/env python3
"""
Generate predictions on test data using a trained PaliGemma LoRA model.

Loads the best checkpoint (deblur + LoRA weights) and runs inference
on the test set, producing a prediction.txt in the format:
    track_id,prediction;confidence

Usage:
    python lora/test.py
    python lora/test.py --checkpoint_dir results/paligemma_lora --test_root data/test
    python lora/test.py --no_deblur --no_4bit
"""

import os
import sys
import argparse
import glob

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm

# Hugging Face imports
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from train module
from lora.train import (
    TrainingConfig,
    PaliGemmaWithDeblur,
    TestDataset,
    get_deblur_module,
    decode_predictions_with_confidence,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PaliGemma LoRA - Test Inference for License Plate OCR"
    )

    # Model settings
    parser.add_argument(
        "--model_name", type=str, default="google/paligemma-3b-pt-224",
        help="Base model name (default: google/paligemma-3b-pt-224)"
    )
    parser.add_argument(
        "--no_4bit", action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--use_8bit", action="store_true",
        help="Use 8-bit quantization instead of 4-bit"
    )

    # Deblur settings
    parser.add_argument(
        "--no_deblur", action="store_true",
        help="Disable deblur module"
    )
    parser.add_argument(
        "--deblur_type", type=str, default="lightweight",
        choices=["lightweight", "esrgan_lite"],
        help="Deblur module type (default: lightweight)"
    )
    parser.add_argument(
        "--flash_attn", action="store_true",
        help="Use SDPA (PyTorch native scaled dot product attention)"
    )

    # Paths
    parser.add_argument(
        "--checkpoint_dir", type=str, default="results/paligemma_lora",
        help="Directory containing the checkpoint (best.pth, deblur_module.pth, paligemma_lora/)"
    )
    parser.add_argument(
        "--test_root", type=str, default="data/test",
        help="Directory containing test data (default: data/test)"
    )

    # Output
    parser.add_argument(
        "-o", "--output", type=str, default="prediction.txt",
        help="Output prediction file name (default: prediction.txt)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save prediction file (default: results/)"
    )

    # Inference settings
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for inference (default: 1)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loader workers (default: 4)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=20,
        help="Maximum number of new tokens to generate (default: 20)"
    )
    parser.add_argument(
        "--img_size", type=int, default=224,
        help="Image size for model input (default: 224)"
    )
    parser.add_argument(
        "--prompt", type=str, default="OCR license plate text:",
        help="Prompt template (default: 'OCR license plate text:')"
    )

    return parser.parse_args()


def load_model(args) -> tuple:
    """Load the trained PaliGemma LoRA model with deblur module.
    
    Returns:
        (model, processor, config)
    """
    config = TrainingConfig(
        model_name=args.model_name,
        use_4bit=not args.no_4bit,
        use_8bit=args.use_8bit,
        use_deblur=not args.no_deblur,
        deblur_type=args.deblur_type,
        use_flash_attn=args.flash_attn,
        img_size=args.img_size,
        prompt_template=args.prompt,
    )

    print(f"Loading processor: {config.model_name}")
    processor = AutoProcessor.from_pretrained(config.model_name, use_fast=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # --- Build base model (VLM + optional deblur) ---
    print(f"Loading base model...")
    model = PaliGemmaWithDeblur(config, processor)

    checkpoint_dir = args.checkpoint_dir

    # --- Load deblur weights ---
    deblur_path = os.path.join(checkpoint_dir, "deblur_module.pth")
    if model.deblur is not None and os.path.exists(deblur_path):
        print(f"Loading deblur weights from: {deblur_path}")
        model.deblur.load_state_dict(
            torch.load(deblur_path, map_location=config.device, weights_only=True)
        )
        model.deblur = model.deblur.to(config.device)
        print("Deblur module loaded.")
    elif model.deblur is not None:
        print(f"WARNING: Deblur weights not found at {deblur_path}, using random init.")
        model.deblur = model.deblur.to(config.device)

    # --- Load LoRA weights ---
    lora_path = os.path.join(checkpoint_dir, "paligemma_lora")
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from: {lora_path}")
        # The model.vlm is already a PeftModel from PaliGemmaWithDeblur.__init__
        # We need to load the saved adapter weights
        model.vlm.load_adapter(lora_path, adapter_name="default")
        print("LoRA adapter loaded.")
    else:
        print(f"WARNING: LoRA weights not found at {lora_path}")

    # --- Load training state (for metadata) ---
    best_ckpt = os.path.join(checkpoint_dir, "best.pth")
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        print(f"Best checkpoint: epoch {state.get('epoch', '?')}, "
              f"best_acc {state.get('best_acc', '?'):.4f}")

    model.eval()
    return model, processor, config


@torch.no_grad()
def run_inference(model, processor, config, args):
    """Run inference on test set and return results."""

    # Create test dataset
    test_dataset = TestDataset(
        root_dir=args.test_root,
        processor=processor,
        img_size=args.img_size,
        prompt=args.prompt,
    )

    if len(test_dataset) == 0:
        print("ERROR: Test dataset is empty!")
        sys.exit(1)

    print(f"Found {len(test_dataset)} test samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    results = []

    for batch in tqdm(test_loader, desc="Inference"):
        pixel_values = batch['pixel_values'].to(config.device, dtype=torch.bfloat16)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        track_ids = batch['track_id']  # list of strings

        # Generate predictions with confidence
        preds_with_conf = decode_predictions_with_confidence(
            model, pixel_values, input_ids, attention_mask,
            processor, max_new_tokens=args.max_new_tokens,
        )

        for i, (pred_text, conf) in enumerate(preds_with_conf):
            results.append((track_ids[i], pred_text, conf))

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Validate paths
    if not os.path.exists(args.test_root):
        print(f"ERROR: Test directory not found: {args.test_root}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PaliGemma LoRA - Test Inference")
    print("=" * 60)
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Test data:      {args.test_root}")
    print(f"Output:         {os.path.join(args.output_dir, args.output)}")
    print(f"Batch size:     {args.batch_size}")
    print("=" * 60)

    # Load model
    model, processor, config = load_model(args)

    # Run inference
    print(f"\nRunning inference on test data...")
    results = run_inference(model, processor, config, args)

    # Sort by track_id for consistent output
    results.sort(key=lambda x: x[0])

    # Format: track_id,prediction;confidence
    submission_data = [
        f"{track_id},{pred_text};{conf:.4f}"
        for track_id, pred_text, conf in results
    ]

    output_path = os.path.join(args.output_dir, args.output)
    with open(output_path, 'w') as f:
        f.write("\n".join(submission_data))

    print(f"\nSaved {len(submission_data)} predictions to {output_path}")

    # Show sample predictions
    print("\nSample predictions:")
    for track_id, pred_text, conf in results[:10]:
        print(f"  {track_id}: {pred_text} (conf: {conf:.4f})")

    # Summary statistics
    confidences = [c for _, _, c in results]
    print(f"\nConfidence stats:")
    print(f"  Mean:   {np.mean(confidences):.4f}")
    print(f"  Median: {np.median(confidences):.4f}")
    print(f"  Min:    {np.min(confidences):.4f}")
    print(f"  Max:    {np.max(confidences):.4f}")


if __name__ == "__main__":
    main()
