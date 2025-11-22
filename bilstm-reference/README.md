# Model 3: LSTM Variants (Reference Implementations)

## Description
Collection of LSTM variant implementations for HAR, including standard LSTM, BiLSTM, and Residual LSTM architectures.

## Source
- **Repository**: `sidharthgurbani/HAR-using-PyTorch`
- **Original Path**: `LSTM/` and `Bidir_Res_LSTM/`

## Architecture Variants
1. **LSTM1/LSTM2** - Standard LSTM (1-2 layers)
2. **Bidir_LSTM1/LSTM2** - Bidirectional LSTM (1-2 layers)
3. **Res_LSTM** - LSTM with residual connections
4. **Res_Bidir_LSTM** - BiLSTM with residual connections

## Key Files
- `LSTM/main.py` - Training script
- `LSTM/model.py` - Model architectures
- `LSTM/config.py` - Architecture configurations
- `LSTM/data_file.py` - Dataset paths
- `Bidir_Res_LSTM/` - Residual BiLSTM implementation

## Configuration
Models selected via `config.py`:
```python
arch = Architecture['Bidir_LSTM1']  # Change architecture here
```

## Usage
```bash
cd LSTM
python main.py
```

## Purpose
Provides base BiLSTM implementation used in Model 4.

## Status
✅ Original implementations (reference only)
