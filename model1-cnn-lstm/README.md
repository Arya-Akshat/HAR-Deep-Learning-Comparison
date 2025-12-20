# Model 1: CNN-LSTM Baseline 🥉

## Results
- **Test Accuracy: 93.08%**
- **F1-Score: 0.9309**
- **Parameters: 209,478**

## Quick Start
```bash
python train_3070ti.py
```
Results saved to `../results/model1-cnn-lstm/`

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
Tanh → Flatten
    ↓
FC(128*128 → 6)
```

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
| WALKING | 0.99 | 1.00 | 0.99 |
| WALKING_UPSTAIRS | 0.92 | 0.94 | 0.93 |
| WALKING_DOWNSTAIRS | 0.95 | 1.00 | 0.97 |
| SITTING | 0.87 | 0.79 | 0.83 |
| STANDING | 0.86 | 0.89 | 0.87 |
| LAYING | 1.00 | 0.97 | 0.98 |

## Files
| File | Description |
|------|-------------|
| `train_3070ti.py` | **Optimized training script** (use this) |
| `network.py` | Model architecture |
| `data_preprocess.py` | Original data loading |
| `main_pytorch.py` | Original training script |
| `Train-CNN-LSTM.ipynb` | Jupyter notebook |

## Source
Originally from `LizLicense/HAR-CNN-LSTM-ATT-pyTorch`
