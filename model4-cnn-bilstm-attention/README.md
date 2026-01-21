# Model 4: CNN-BiLSTM-Attention 🥈

## Results
- **Test Accuracy: 93.38%**
- **F1-Score: 0.9350**
- **Parameters: 187,654**

## Quick Start
```bash
python train_3070ti.py
```
Outputs saved to `../results/model4-cnn-bilstm-attention/` (checkpoint: `best_model.pth`). Latest metrics live in `../results/TRAINING_LOG.md`.

## Architecture
```
Input (batch, 128, 9)
    ↓
BiLSTM Layer 1 (128 hidden, bidirectional)
    ↓
BiLSTM Layer 2 (128 hidden, bidirectional)
    ↓
Dropout(0.5)
    ↓
Temporal Attention → Context Vector
    ↓
FC → 6 Classes
```

## Key Insight
BiLSTM's bidirectional context captures temporal patterns better than unidirectional LSTM. Combined with attention, achieves best accuracy-to-parameter ratio.

## Configuration
| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 0.0015 |
| Epochs | 100 |
| Dropout | 0.5 |
| Hidden size | 128 |
| LSTM layers | 2 |
| Optimizer | Adam |

## Per-Class Performance
| Activity | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| WALKING | 1.00 | 0.97 | 0.98 |
| WALKING_UPSTAIRS | 0.97 | 0.97 | 0.97 |
| WALKING_DOWNSTAIRS | 0.96 | 1.00 | 0.98 |
| SITTING | 0.83 | 0.85 | 0.84 |
| STANDING | 0.90 | 0.83 | 0.86 |
| LAYING | 0.95 | 1.00 | 0.98 |

## Files
| File | Description |
|------|-------------|
| `train_3070ti.py` | **Optimized training script** (use this) |
| `attention.py` | TemporalAttn class |
| `model.py` | BiLSTM architecture |
| `config.py` | Model configurations |
| `main.py` | Original training script |

## Source
Combined from:
- `sidharthgurbani/HAR-using-PyTorch` (BiLSTM base)
- `LizLicense/HAR-CNN-LSTM-ATT-pyTorch` (Attention)
