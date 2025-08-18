# UMAFall Multi-Task Learning Pipeline

A comprehensive PyTorch implementation of a multi-task learning pipeline for simultaneous fall detection and activity recognition using the UMAFall wearable sensor dataset.

## Features

- **Multi-Task Learning**: Joint optimization of fall detection (binary) and ADL recognition (multi-class)
- **Multiple Architectures**: CNN-BiLSTM, TCN, and Lite Transformer backbones
- **Advanced Loss Functions**: Uncertainty weighting, GradNorm, and focal loss
- **Robust Evaluation**: Cross-subject validation with LOSO and k-fold splits
- **Comprehensive Analysis**: Error analysis, McNemar tests, and per-activity metrics
- **Production Ready**: Deterministic training, extensive testing, and validation

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/umafall-mtl-pipeline.git
cd umafall-mtl-pipeline

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Data Preparation

1. Download the UMAFall dataset and extract to `data/raw/UMAFall_Dataset/`

2. Run data preparation:
```bash
python scripts/prepare_data.py --config configs/dataset.yaml
```

This will:
- Discover and validate all CSV files
- Create sliding windows with configurable size/stride
- Generate cross-subject splits (k-fold and LOSO)
- Compute normalization statistics
- Save processed data and manifests

### Training

Train a model with default configuration:
```bash
python scripts/train.py --config configs/train.yaml
```

Train specific experiments:
```bash
# Single-task baselines
python scripts/train.py --config configs/experiments/mvs_baseline_st_fall.yaml
python scripts/train.py --config configs/experiments/mvs_baseline_st_adl.yaml

# Multi-task with uncertainty weighting
python scripts/train.py --config configs/experiments/mvs_mtl_uncertainty.yaml

# Multi-task with GradNorm
python scripts/train.py --config configs/experiments/mvs_mtl_gradnorm.yaml
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --split test \
    --output reports/
```

### Error Analysis

Compare models and analyze errors:
```bash
python scripts/analyze_errors.py \
    --checkpoint checkpoints/mtl_model.pt \
    --baseline checkpoints/baseline_model.pt \
    --output reports/error_analysis/
```

### Run All Experiments

Execute the complete experiment suite:
```bash
bash scripts/run_experiments.sh
```

## Project Structure

```
umafall-mtl-pipeline/
├── configs/              # Configuration files
│   ├── dataset.yaml     # Dataset configuration
│   ├── model.yaml       # Model architecture config
│   ├── train.yaml       # Training configuration
│   └── experiments/     # Experiment configs
├── data_ingest/         # Data loading and discovery
├── preprocessing/       # Normalization, windowing, augmentation
├── splits/              # Cross-subject splitting
├── models/              # Model architectures
├── training/            # Training utilities
├── eval/                # Evaluation metrics
├── analysis/            # Error analysis tools
├── scripts/             # Main execution scripts
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Configuration

### Dataset Configuration

Edit `configs/dataset.yaml`:
```yaml
dataset:
  root_dir: "data/raw/UMAFall_Dataset"
  sampling_rate: 50  # Hz
  window_size: 128    # samples (2.56s @ 50Hz)
  stride: 64          # 50% overlap
```

### Model Configuration

Edit `configs/model.yaml`:
```yaml
model:
  backbone: "cnn_bilstm"  # Options: cnn_bilstm, tcn, lite_transformer
  input_channels: 6       # 3 acc + 3 gyro
  use_magnitude: false    # Add magnitude channel
```

### Training Configuration

Edit `configs/train.yaml`:
```yaml
training:
  epochs: 100
  batch_size: 64
  learning_rate: 1e-3
  loss_type: "gradnorm"  # Options: weighted, uncertainty, gradnorm
  use_balanced_sampler: true
  fall_ratio: 0.15       # Min ratio of fall samples per batch
```

## Experiments

### Minimal Viable Set (MVS)

1. **ST-Fall**: Single-task fall detection baseline
2. **ST-ADL**: Single-task activity recognition baseline
3. **MTL-Shared**: Multi-task with fixed weights (0.5/0.5)
4. **MTL-Uncertainty**: Multi-task with learnable uncertainty weights
5. **MTL-GradNorm**: Multi-task with gradient normalization

### Full Experiment Grid

- Task weight variations: (0.3/0.7), (0.5/0.5), (0.7/0.3)
- Loss functions: Weighted BCE vs Focal loss (γ ∈ {1, 2})
- Robustness: LOSO evaluation, window size ablation
- Architecture comparison: CNN-BiLSTM vs TCN vs Transformer

## Metrics

### Fall Detection
- Precision, Recall, F1-score
- PR-AUC (Precision-Recall Area Under Curve)
- False alarms per hour
- Optimized threshold using F-β score (β=2)

### Activity Recognition
- Accuracy
- Macro and weighted F1-scores
- Per-class F1-scores
- Confusion matrix

### Multi-Task
- Composite metric: weighted sum of fall PR-AUC and ADL macro-F1
- Per-task loss tracking
- Task weight evolution (for adaptive methods)

## Testing

Run unit tests:
```bash
pytest tests/ -v --cov=.
```

Verify installation and data:
```bash
python scripts/verify_run.py
```

## Reproducibility

- Fixed random seeds across NumPy, PyTorch, and CUDA
- Deterministic operations enabled
- Git hash and configuration logged with checkpoints
- Comprehensive configuration management with OmegaConf

## Citations

If you use this code, please cite:

```bibtex
@inproceedings{umafall2023,
  title={Multi-Task Learning for Wearable Fall Detection and Activity Recognition},
  author={Your Name},
  booktitle={Conference Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- UMAFall dataset creators
- PyTorch and scikit-learn communities
- Anthropic for Claude's assistance in development