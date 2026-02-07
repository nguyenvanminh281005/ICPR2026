# ICPR2026

Multi-frame OCR solution for the **ICPR 2026 Challenge on Low-Resolution License Plate Recognition**.

This implementation combines temporal information from 5 video frames using attention fusion mechanisms to achieve robust recognition on low-resolution license plates.

🔗 **Challenge:** [ICPR 2026 LRLPR](https://icpr26lrlpr.github.io/)

---

## Quick Start

```bash
# Install dependencies
uv sync

# Train with default settings (ResTranOCR + STN)
python train.py

# Train CRNN baseline
python train.py --model crnn --experiment-name crnn_baseline

# Generate submission file
python train.py --submission-mode --model restran
```

---

## Key Features

- **Multi-Frame Fusion**: Processes 5-frame sequences with attention-based fusion
- **Spatial Transformer Network**: Optional STN module for automatic image alignment
- **Dual Architectures**: CRNN (baseline) and ResTranOCR (ResNet34 + Transformer)
- **Smart Data Augmentation**: Scenario-B aware validation split with configurable augmentation levels
- **Production Ready**: Mixed precision training, gradient clipping, OneCycleLR scheduler

---

## Model Architectures

### CRNN (Baseline)
**Pipeline:** Multi-frame Input → STN Alignment → CNN → Attention Fusion → BiLSTM → CTC

Simple and effective baseline using convolutional features and bidirectional LSTM for sequence modeling.

### ResTranOCR (Advanced)
**Pipeline:** Multi-frame Input → STN Alignment → ResNet34 → Attention Fusion → Transformer → CTC

Modern architecture leveraging ResNet34 backbone and Transformer encoder with positional encoding for improved long-range dependencies.

**Both models accept input shape:** `(Batch, 5, 3, 32, 128)` and output character sequences via CTC decoding.

---

## Installation

**Requirements:**
- Python 3.11+
- CUDA-enabled GPU (recommended)

**Using uv (recommended):**
```bash
git clone https://github.com/nguyenvanminh281005/ICPR2026
cd ICPR2026
uv sync
```

**Using pip:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install albumentations opencv-python matplotlib numpy pandas tqdm
```

---

## Usage

### Data Preparation

Organize your dataset with the following structure:

```
data/train/
├── track_001/
│   ├── lr-001.png
│   ├── lr-002.png
│   ├── ...
│   ├── hr-001.png (optional, for synthetic LR generation)
│   └── annotations.json
└── track_002/
    └── ...
```

**annotations.json format:**
```json
{"plate_text": "ABC1234"}
```

### Training

**Basic training:**
```bash
python train.py
```

**Custom configuration:**
```bash
python train.py \
    --model restran \
    --experiment-name my_experiment \
    --data-root /path/to/dataset \
    --batch-size 64 \
    --epochs 30 \
    --lr 0.0005 \
    --aug-level full
```

**Disable STN:**
```bash
python train.py --no-stn
```

**Key arguments:**
- `-m, --model`: Model type (`crnn` or `restran`)
- `-n, --experiment-name`: Experiment identifier
- `--data-root`: Path to training data (default: `data/train`)
- `--batch-size`: Batch size (default: 64)
- `--epochs`: Training epochs (default: 30)
- `--lr`: Learning rate (default: 5e-4)
- `--aug-level`: Augmentation level (`full` or `light`)
- `--no-stn`: Disable Spatial Transformer Network
- `--submission-mode`: Train on full dataset and generate test predictions
- `--output-dir`: Output directory (default: `results/`)

### Ablation Studies

Run automated experiments comparing different configurations:

```bash
python run_ablation.py
```

Experiments:
- CRNN with/without STN
- ResTranOCR with/without STN

Results saved in `experiments/ablation_summary.txt`.

### Outputs

After training, the following files are generated in the output directory:

- `{experiment_name}_best.pth` - Best model checkpoint
- `submission_{experiment_name}.txt` - Predictions in competition format: `track_id,predicted_text;confidence`

---

## Configuration

Key hyperparameters in `configs/config.py`:

```python
MODEL_TYPE = "restran"           # "crnn" or "restran"
USE_STN = True                   # Enable/disable STN
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EPOCHS = 30
AUGMENTATION_LEVEL = "full"      # "full" or "light"

# CRNN specific
HIDDEN_SIZE = 256
RNN_DROPOUT = 0.25

# ResTranOCR specific
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 3
TRANSFORMER_FF_DIM = 2048
TRANSFORMER_DROPOUT = 0.1
```

All config parameters can be overridden via CLI arguments.

---

## Project Structure

```
.
├── configs/
│   └── config.py              # Configuration dataclass
├── src/
│   ├── data/
│   │   ├── dataset.py         # MultiFrameDataset with scenario-aware splitting
│   │   └── transforms.py      # Augmentation pipelines
│   ├── models/
│   │   ├── crnn.py            # CRNN baseline
│   │   ├── restran.py         # ResTranOCR advanced model
│   │   └── components.py      # Shared modules (STN, AttentionFusion, etc.)
│   ├── training/
│   │   └── trainer.py         # Training loop and validation
│   └── utils/
│       ├── common.py          # Utility functions
│       └── postprocess.py     # CTC decoding
├── train.py                   # Main training script
├── run_ablation.py            # Ablation study automation
└── pyproject.toml             # Dependencies
```

---

## Technical Details

### Attention Fusion Module
Dynamically computes attention weights across temporal frames and fuses multi-frame features into a single representation before sequence modeling.

### Data Augmentation
- **Full mode**: Affine transforms, perspective warping, HSV adjustment, coarse dropout
- **Light mode**: Resize and normalize only
- **Scenario-B aware splitting**: Validation set prioritizes challenging scenarios to prevent overfitting