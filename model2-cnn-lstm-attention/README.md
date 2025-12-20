# Model 2: CNN-LSTM-Attention (4th Place)

## Results
- **Test Accuracy: 92.64%**
- **F1-Score: 0.9266**
- **Parameters: 161,094**

⚠️ **Finding**: Attention did NOT improve over baseline (92.64% vs 93.08%)

## Quick Start
```bash
python train_3070ti.py
```
Results saved to `../results/model2-cnn-lstm-attention/`

## Architecture
```
Input (batch, 9, 128)
    ↓
Conv1D(9→64, k=6) + ReLU + MaxPool
    ↓
Conv1D(64→128, k=3) + ReLU + MaxPool
    ↓
Dropout(0.1)
    ↓
LSTM(input=32, hidden=128)
    ↓
Tanh
    ↓
Temporal Attention → Context Vector (128)
    ↓
FC(128 → 6)
```

## Temporal Attention
The attention mechanism computes:
- Query: learned parameter
- Keys/Values: LSTM hidden states
- Output: weighted sum of hidden states

This reduces FC input from `128*128` to `128`, fewer parameters but lower accuracy.

## Configuration
| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 0.001 |
| Epochs | 50 |
| Dropout | 0.1 |
| Optimizer | Adam |

## Per-Class Performance
| Activity | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| WALKING | 0.99 | 0.94 | 0.96 |
| WALKING_UPSTAIRS | 0.93 | 0.97 | 0.95 |
| WALKING_DOWNSTAIRS | 0.97 | 1.00 | 0.98 |
| SITTING | 0.88 | 0.75 | 0.81 |
| STANDING | 0.83 | 0.91 | 0.87 |
| LAYING | 0.96 | 1.00 | 0.98 |

## Files
| File | Description |
|------|-------------|
| `train_3070ti.py` | **Optimized training script** (use this) |
| `attention.py` | TemporalAttn class |
| `network.py` | Model architecture |
| `data_preprocess.py` | Original data loading |
| `main_pytorch.py` | Original training script |

## Source
Originally from `LizLicense/HAR-CNN-LSTM-ATT-pyTorch`
