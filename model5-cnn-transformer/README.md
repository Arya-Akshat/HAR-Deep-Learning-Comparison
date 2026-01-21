# Model 5: CNN-Transformer Ultimate 🥇

## Results
- **Test Accuracy: 93.48%** (Best!)
- **F1-Score: 0.9341**
- **Parameters: 2,359,494**

## Quick Start
```bash
# Best results (recommended)
python train_ultimate.py

# Standard optimized version
python train_3070ti.py
```
Outputs saved to `../results/model5-cnn-transformer/` (checkpoint: `best_model.pth`). Latest metrics live in `../results/TRAINING_LOG.md`.

## Architecture
```
Input (batch, 128, 9)
    ↓
Conv1D Projection (9 → 256) with GELU
    ↓
CLS Token Prepend
    ↓
Positional Encoding
    ↓
Transformer Encoder (4 layers, 8 heads)
    ↓
CLS Token Extract
    ↓
MLP Head → 6 Classes
```

## Ultimate Configuration
| Parameter | Value |
|-----------|-------|
| Transformer dim | 256 |
| Attention heads | 8 |
| Encoder layers | 4 |
| Feed-forward dim | 512 |
| Dropout | 0.3 |
| Batch size | 64 |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Label smoothing | 0.1 |
| Warmup epochs | 20 |
| Max epochs | 300 |
| Early stopping | 50 epochs patience |

## Key Optimizations
- ✅ Warmup scheduler (critical for transformers)
- ✅ Label smoothing (reduces overconfidence)
- ✅ AdamW with weight decay
- ✅ Gradient clipping
- ✅ Larger model (256 dim vs 64)
- ✅ Cosine annealing LR

## Per-Class Performance
| Activity | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| WALKING | 0.96 | 0.96 | 0.96 |
| WALKING_UPSTAIRS | 0.98 | 0.93 | 0.95 |
| WALKING_DOWNSTAIRS | 0.90 | 0.96 | 0.93 |
| SITTING | 0.91 | 0.83 | 0.87 |
| STANDING | 0.89 | 0.93 | 0.91 |
| LAYING | 0.96 | 1.00 | 0.98 |

## Config Files
| File | Description |
|------|-------------|
| `config_ultimate.json` | Best results config (2.3M params) |
| `config_3070ti_optimized.json` | Standard optimized (600K params) |
| `config_uci.json` | Original UCI config |
| `config.json` | Original paper config |

## Files
| File | Description |
|------|-------------|
| `train_ultimate.py` | **Best results** (use this) |
| `train_3070ti.py` | Standard optimized training |
| `models/IMUTransformerEncoder.py` | Transformer architecture |
| `main.py` | Original training script |

## Source
Based on "Boosting Inertial-based Human Activity Recognition with Transformers" (Shavit and Klein, 2021)
- Paper: IEEE Open Access
- Original repo structure retained in `models/` and `util/`
