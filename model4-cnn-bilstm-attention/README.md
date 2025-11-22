# Model 4: CNN-BiLSTM-Attention (Our Implementation)

## Description
**Main contribution**: Combines BiLSTM architecture with temporal attention mechanism for improved HAR performance.

## Components Combined
1. **BiLSTM Base** from `sidharthgurbani/HAR-using-PyTorch`
2. **Attention Mechanism** from `LizLicense/HAR-CNN-LSTM-ATT-pyTorch`

## Architecture
```
Input (9 features, 128 timesteps)
    ↓
BiLSTM Layer 1 (bidirectional)
    ↓
Highway BiLSTM Layers
    ↓
Dropout
    ↓
Temporal Attention → Context Vector
    ↓
Fully Connected → 6 Classes
```

## Key Files
- `attention.py` - TemporalAttn class (from Model 2)
- `model.py` - Bidir_LSTM_Attention_Model class
- `config.py` - Model configuration (set as active)
- `main.py` - Training script with attention model selection
- `data_file.py` - UCI-HAR dataset paths
- `train.py`, `test.py`, `Functions.py` - Training utilities
- `loadDataset.py` - Data loading functions

## Dataset Configuration
- UCI-HAR dataset path configured in `data_file.py`
- 7,352 training samples
- 2,947 test samples
- 9 input features (accel + gyro)
- 6 activity classes

## Virtual Environment
Location: `../HAR-using-PyTorch/.venv`
Created with UV for fast dependency management.

## Installation
```bash
# Create virtual environment
uv venv .venv --python 3.9
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision torchaudio numpy pandas matplotlib scikit-learn seaborn
```

## Usage
```bash
source .venv/bin/activate
python main.py
```

## Configuration
Active architecture in `config.py`:
```python
arch = Architecture['Bidir_LSTM_Attention']
```

## Hyperparameters
- Input size: 9
- Hidden size: 32
- Layers: 2 (bidirectional)
- Dropout: 0.5
- Batch size: 64
- Learning rate: 0.0015
- Epochs: 120

## Expected Performance (UCI-HAR)
- Test Accuracy: ~93-95%
- F1-Score: ~0.94

## Results Location
- Metrics: `results/results_bidir_lstm_attention.txt`
- Plots: `results/*.png`

## Status
✅ **Implementation Complete - Ready for Training**

## Implementation Date
November 22, 2025

## Related Documentation
- See `CNN-BiLSTM-Attention-Implementation.ipynb` for detailed implementation guide
- See `README_BiLSTM_Attention.md` for comprehensive documentation
