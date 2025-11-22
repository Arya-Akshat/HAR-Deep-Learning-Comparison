# Model 2: CNN-LSTM-Attention (Proposed)

## Description
CNN-LSTM model enhanced with temporal attention mechanism.

## Source
- **Repository**: `LizLicense/HAR-CNN-LSTM-ATT-pyTorch`
- **Original Path**: `GitFYP_experiment/supervised/UCI/Attention/`

## Architecture
- Conv1D layers for feature extraction
- LSTM for temporal modeling
- **Temporal Attention layer** - focuses on important timesteps
- Fully connected classifier

## Key Files
- `main_pytorch.py` - Training script
- `network.py` - Model with attention
- `attention.py` - TemporalAttn class
- `data_preprocess.py` - Data loading

## Key Innovation
Attention mechanism reduces FC input from `128*128` to `128` by attending to important timesteps.

## Usage
```bash
python main_pytorch.py --nepoch 50 --batchsize 64
```

## Expected Performance (UCI-HAR)
- Test Accuracy: ~92-94%
- F1-Score: ~0.93

## Status
✅ Already implemented and tested
