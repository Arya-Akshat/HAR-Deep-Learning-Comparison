# HAR Model Testing Scripts

This folder contains testing scripts for each trained model. Each script loads the corresponding best model from the `results/` folder and generates confusion matrices.

## Individual Model Tests

| Script | Model | Description |
|--------|-------|-------------|
| `test_model1_cnn_lstm.py` | CNN-LSTM | Baseline CNN-LSTM model |
| `test_model2_cnn_lstm_attention.py` | CNN-LSTM-Attention | CNN-LSTM with Temporal Attention |
| `test_model4_bilstm_attention.py` | BiLSTM-Attention | Bidirectional LSTM with Attention |
| `test_model5_cnn_transformer.py` | CNN-Transformer | Transformer encoder with CNN input projection |

## Usage

### Run Individual Model Test
```bash
python test_model1_cnn_lstm.py
python test_model2_cnn_lstm_attention.py
python test_model4_bilstm_attention.py
python test_model5_cnn_transformer.py
```

### Run All Tests at Once
```bash
python run_all_tests.py
```

## Output Files

Each test generates:
- `confusion_matrix_model{N}_*.png` - Raw confusion matrix
- `confusion_matrix_model{N}_*_normalized.png` - Row-normalized confusion matrix

Running `run_all_tests.py` additionally generates:
- `model_comparison_chart.png` - Bar chart comparing accuracy and F1 scores
- `all_confusion_matrices_combined.png` - All 4 confusion matrices in one figure

## Model Configurations

All configurations are **EXACT** copies from the training scripts to ensure reproducibility:

### Model 1: CNN-LSTM
- Batch size: 64
- Dropout: 0.1
- Input shape: (batch, 9, 128) - channels first

### Model 2: CNN-LSTM-Attention
- Batch size: 64
- Dropout: 0.1
- LSTM hidden size: 128
- Input shape: (batch, 9, 128) - channels first

### Model 4: BiLSTM-Attention
- Batch size: 64
- Dropout: 0.5
- Hidden size: 128
- LSTM layers: 2 (bidirectional)
- Input shape: (batch, 128, 9) - sequence first

### Model 5: CNN-Transformer (Ultimate)
- Transformer dim: 256
- Attention heads: 8
- Encoder layers: 4
- Feed-forward dim: 512
- Dropout: 0.3
- Input shape: (batch, 128, 9) - sequence first

## Requirements

- PyTorch
- NumPy
- scikit-learn
- matplotlib
- seaborn
