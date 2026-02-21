#!/usr/bin/env python3
"""Generate predictions on test data using a trained model."""
import argparse
import os
import sys
from collections import Counter
from itertools import product

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.utils.postprocess import decode_with_confidence


def vote_predictions(predictions):
    """Character-level voting across multiple predictions.

    Args:
        predictions: List of (text, confidence) tuples from different frame combinations.

    Returns:
        (voted_text, average_confidence)
    """
    if not predictions:
        return "", 0.0

    texts = [p[0] for p in predictions]
    confs = [p[1] for p in predictions]

    # Determine the most common prediction length
    length_counts = Counter(len(t) for t in texts)
    target_length = length_counts.most_common(1)[0][0]

    # Filter to predictions with the most common length for cleaner voting
    filtered_texts = [t for t in texts if len(t) == target_length]
    filtered_confs = [c for t, c in zip(texts, confs) if len(t) == target_length]

    if not filtered_texts:
        # Fallback: use all predictions
        filtered_texts = texts
        filtered_confs = confs
        target_length = max(len(t) for t in texts) if texts else 0

    # Vote character by character
    voted_chars = []
    for pos in range(target_length):
        char_counts = Counter()
        for t in filtered_texts:
            if pos < len(t):
                char_counts[t[pos]] += 1
        if char_counts:
            voted_chars.append(char_counts.most_common(1)[0][0])

    avg_conf = sum(filtered_confs) / len(filtered_confs) if filtered_confs else 0.0
    return "".join(voted_chars), avg_conf


def ensemble_inference(model, dataset, config, device, ensemble_batch_size=32):
    """Run ensemble inference using paired frame combinations per track.

    For each test-public track (10 frames), forms 5 pairs: (0,5), (1,6),
    (2,7), (3,8), (4,9). For each pair, picks one of the two frames,
    giving 2^5 = 32 combinations. Runs prediction on each and votes
    character-by-character to produce the final prediction.

    Args:
        model: Trained model.
        dataset: MultiFrameDataset with ensemble=True (returns all frames).
        config: Config object.
        device: torch device.
        ensemble_batch_size: How many combinations to batch at once.

    Returns:
        List of (track_id, pred_text, confidence) tuples.
    """
    results = []

    for idx in tqdm(range(len(dataset)), desc="Ensemble Inference"):
        images_tensor, _, _, _, track_id = dataset[idx]
        # images_tensor shape: [N, C, H, W] where N can be 5 or 10
        num_frames = images_tensor.shape[0]

        if num_frames <= 5:
            # Standard single prediction (no ensemble needed)
            input_tensor = images_tensor.unsqueeze(0).to(device)  # [1, 5, C, H, W]
            with torch.no_grad():
                preds = model(input_tensor)
            decoded = decode_with_confidence(preds, config.IDX2CHAR)
            results.append((track_id, decoded[0][0], decoded[0][1]))
            continue

        # Generate 2^5 = 32 combinations from 5 pairs: (0,5), (1,6), (2,7), (3,8), (4,9)
        pairs = [(i, i + 5) for i in range(5)]  # [(0,5), (1,6), (2,7), (3,8), (4,9)]
        combos = [tuple(pair[choice] for pair, choice in zip(pairs, choices))
                  for choices in product(range(2), repeat=5)]

        all_predictions = []

        # Process combinations in batches
        for batch_start in range(0, len(combos), ensemble_batch_size):
            batch_combos = combos[batch_start:batch_start + ensemble_batch_size]
            batch_tensors = []
            for combo in batch_combos:
                frames = images_tensor[list(combo)]  # [5, C, H, W]
                batch_tensors.append(frames)

            batch = torch.stack(batch_tensors, 0).to(device)  # [B, 5, C, H, W]
            with torch.no_grad():
                preds = model(batch)
            decoded = decode_with_confidence(preds, config.IDX2CHAR)
            all_predictions.extend(decoded)

        # Vote across all combination predictions
        voted_text, voted_conf = vote_predictions(all_predictions)
        results.append((track_id, voted_text, voted_conf))

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate predictions on test data"
    )
    parser.add_argument(
        "-m", "--model-path", type=str,
        default="results/restran_best.pth",
        help="Path to trained model checkpoint (default: results/restran_best.pth)"
    )
    parser.add_argument(
        "--model-type", type=str, choices=["crnn", "restran"],
        default="restran",
        help="Model architecture: 'crnn' or 'restran' (default: restran)"
    )
    parser.add_argument(
        "--test-dir", type=str, default="data/test",
        help="Directory containing test data (default: data/test)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="submission.txt",
        help="Output submission file name (default: submission.txt)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save submission file (default: results/)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for inference (default: 64)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=10,
        help="Number of data loader workers (default: 10)"
    )
    parser.add_argument(
        "--no-stn", action="store_true",
        help="Disable Spatial Transformer Network (must match training config)"
    )
    parser.add_argument(
        "--temporal-fusion", type=str,
        choices=["attention", "quality", "transformer", "hybrid"],
        default=None,
        help="Temporal fusion type (default: from config)"
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Enable ensemble mode for test-public: use 2^5=32 paired frame combinations and vote"
    )
    parser.add_argument(
        "--ensemble-batch-size", type=int, default=32,
        help="Batch size for ensemble combinations (default: 32)"
    )
    return parser.parse_args()


def main():
    """Main inference entry point."""
    args = parse_args()
    
    # Initialize config for model parameters
    config = Config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Model checkpoint not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.test_dir):
        print(f"❌ ERROR: Test directory not found: {args.test_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📦 Loading model from: {args.model_path}")
    print(f"📂 Test data directory: {args.test_dir}")
    print(f"📝 Output file: {os.path.join(args.output_dir, args.output)}")
    
    # Create test dataset
    is_ensemble = args.ensemble
    test_ds = MultiFrameDataset(
        root_dir=args.test_dir,
        mode='val',
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        seed=config.SEED,
        is_test=True,
        ensemble=is_ensemble,
    )
    
    if len(test_ds) == 0:
        print("❌ ERROR: Test dataset is empty!")
        sys.exit(1)
    
    print(f"📊 Found {len(test_ds)} test samples")
    
    # Initialize model
    use_stn = not args.no_stn
    temporal_fusion = args.temporal_fusion if args.temporal_fusion else config.TEMPORAL_FUSION_TYPE
    if args.model_type == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=use_stn,
            temporal_fusion_type=temporal_fusion,
        )
    else:
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=use_stn,
        )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model ({args.model_type}): {total_params:,} parameters")
    
    # Run inference
    if is_ensemble:
        print(f"\n🔮 Running ENSEMBLE inference (2^5=32 paired combinations + character voting)...")
        results = ensemble_inference(
            model, test_ds, config, device,
            ensemble_batch_size=args.ensemble_batch_size,
        )
    else:
        print(f"\n🔮 Running inference on test data...")
        # Create data loader (standard mode)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=MultiFrameDataset.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True
        )
        results = []
        with torch.no_grad():
            for images, _, _, _, track_ids in tqdm(test_loader, desc="Test Inference"):
                images = images.to(device)
                preds = model(images)
                decoded_list = decode_with_confidence(preds, config.IDX2CHAR)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
    
    # Sort by track_id for consistent output
    results.sort(key=lambda x: x[0])
    
    # Format and save submission file
    submission_data = [f"{track_id},{pred_text};{conf:.4f}" for track_id, pred_text, conf in results]
    output_path = os.path.join(args.output_dir, args.output)
    
    with open(output_path, 'w') as f:
        f.write("\n".join(submission_data))
    
    print(f"\n✅ Saved {len(submission_data)} predictions to {output_path}")
    
    # Show sample predictions
    print("\n📋 Sample predictions:")
    for track_id, pred_text, conf in results[:5]:
        print(f"   {track_id}: {pred_text} (conf: {conf:.4f})")


if __name__ == "__main__":
    main()
