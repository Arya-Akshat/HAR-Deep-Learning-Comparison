# Model 1: CNN-LSTM (Baseline)

## Description
Baseline CNN-LSTM model for Human Activity Recognition.

## Source
- **Repository**: `LizLicense/HAR-CNN-LSTM-ATT-pyTorch`
- **Original Path**: `GitFYP_experiment/supervised/UCI/`

## Architecture
- Conv1D layers for feature extraction
- LSTM for temporal modeling
- Fully connected classifier
- No attention mechanism

## Key Files
- `main_pytorch.py` - Training script
- `network.py` - Model architecture
- `data_preprocess.py` - Data loading

## Usage
```bash
python main_pytorch.py --nepoch 50 --batchsize 64
```

## Expected Performance (UCI-HAR)
- Test Accuracy: ~90-92%
- F1-Score: ~0.91

## Status
✅ Already implemented and tested
