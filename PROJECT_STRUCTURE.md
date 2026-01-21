# Human Activity Recognition - Comparative Study

## 📁 Project Structure

This repository contains implementations of 4 different deep learning architectures for Human Activity Recognition (HAR) on the UCI-HAR dataset, organized by model type.

```
HAR-Deep-Learning-Comparison/
├── model1-cnn-lstm/                    # Baseline CNN-LSTM (93.08%) 🥉
├── model2-cnn-lstm-attention/          # CNN-LSTM with Attention (92.64%)
├── model4-cnn-bilstm-attention/        # BiLSTM-Attention (93.38%) 🥈
├── model5-cnn-transformer/             # Transformer (93.48%) 🥇
├── bilstm-reference/                   # Reference BiLSTM implementations
│
├── results/                            # Training outputs & checkpoints
│   ├── TRAINING_LOG.md                 # Authoritative metrics
│   ├── model1-cnn-lstm/                # Model 1 outputs
│   ├── model2-cnn-lstm-attention/      # Model 2 outputs
│   ├── model4-cnn-bilstm-attention/    # Model 4 outputs
│   └── model5-cnn-transformer/         # Model 5 outputs
│
├── human+activity+recognition+using+smartphones/  # UCI-HAR Dataset
├── README.md                           # Main documentation
├── RESULTS_COMPARISON.md               # Detailed analysis
└── PROJECT_STRUCTURE.md                # This file
```

## 🎯 Models Overview

| Rank | Folder | Model | Accuracy | F1-Score | Parameters |
|------|--------|-------|----------|----------|------------|
| 🥇 | `model5-cnn-transformer/` | CNN-Transformer | **93.48%** | 0.9341 | 2,359K |
| 🥈 | `model4-cnn-bilstm-attention/` | BiLSTM-Attention | 93.38% | **0.9350** | 188K |
| 🥉 | `model1-cnn-lstm/` | CNN-LSTM (Baseline) | 93.08% | 0.9309 | 209K |
| 4th | `model2-cnn-lstm-attention/` | CNN-LSTM-Attention | 92.64% | 0.9266 | 161K |

## 📊 Model Details

### Model 1: CNN-LSTM (Baseline) 🥉
- **Path**: `model1-cnn-lstm/`
- **Accuracy**: 93.08%
- **Architecture**: Conv1D + LSTM + FC
- **Status**: ✅ Complete

### Model 2: CNN-LSTM-Attention
- **Path**: `model2-cnn-lstm-attention/`
- **Accuracy**: 92.64%
- **Architecture**: Conv1D + LSTM + Temporal Attention + FC
- **Finding**: ⚠️ Attention decreased performance vs baseline
- **Status**: ✅ Complete

### Model 4: BiLSTM-Attention 🥈
- **Path**: `model4-cnn-bilstm-attention/`
- **Accuracy**: 93.38%
- **Architecture**: BiLSTM (2 layers) + Temporal Attention + FC
- **Key Insight**: Best efficiency - 93.38% with only 188K parameters
- **Status**: ✅ Complete

### Model 5: CNN-Transformer 🥇
- **Path**: `model5-cnn-transformer/`
- **Accuracy**: 93.48% (Best!)
- **Architecture**: Conv1D + Transformer Encoder (4 layers, 8 heads) + MLP
- **Status**: ✅ Complete

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.9+
# UV package manager (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Training Model 4 (Our Implementation)
```bash
cd model4-cnn-bilstm-attention

# Create virtual environment
uv venv .venv --python 3.9
source .venv/bin/activate

# Install dependencies
uv pip install torch numpy pandas matplotlib scikit-learn seaborn

# Run training
python main.py
```

### Training Models 1 & 2
```bash
cd model1-cnn-lstm/UCI/  # or model2-cnn-lstm-attention/Attention/
python main_pytorch.py --nepoch 50 --batchsize 64
```

## 📈 Expected Results

### Performance Comparison (UCI-HAR Dataset)

| Model | Test Accuracy | F1-Score | Parameters |
|-------|--------------|----------|------------|
| Model 1 (CNN-LSTM) | 93.08% | 0.931 | 209K |
| Model 2 (CNN-LSTM-Att) | 92.64% | 0.927 | 161K |
| **Model 4 (BiLSTM-Att)** | **93.38%** | **0.935** | **188K** |
| Model 5 (Transformer) | **93.48%** 🏆 | 0.934 | 2,359K |

## 📚 Dataset

### UCI-HAR Dataset
- **Location**: `human+activity+recognition+using+smartphones/UCI HAR Dataset/`
- **Train samples**: 7,352 sequences
- **Test samples**: 2,947 sequences
- **Sequence length**: 128 timesteps
- **Input features**: 9 channels
  - 3-axis body accelerometer
  - 3-axis body gyroscope
  - 3-axis total accelerometer
- **Activities**: 6 classes
  1. WALKING
  2. WALKING_UPSTAIRS
  3. WALKING_DOWNSTAIRS
  4. SITTING
  5. STANDING
  6. LAYING

## 📝 Documentation

### Model-Specific README Files
- `model1-cnn-lstm/README.md` - Baseline model documentation
- `model2-cnn-lstm-attention/README.md` - Attention model documentation
- `model4-cnn-bilstm-attention/README.md` - BiLSTM-Attention details
- `model5-cnn-transformer/README.md` - Transformer documentation
- `bilstm-reference/README.md` - Reference implementations

### Results & Authoritative Metrics
- **`results/TRAINING_LOG.md`** - Source of truth for all metrics
- Each model folder in `results/` contains:
  - `best_model.pth` - Trained model checkpoint
  - `confusion_matrix.png` - Per-class performance visualization
  - `training_curves.png` - Loss and accuracy over epochs

## 🔗 Source Repositories

The models in this project were adapted from:
1. [HAR-CNN-LSTM-ATT-pyTorch](https://github.com/LizLicense/HAR-CNN-LSTM-ATT-pyTorch) - Models 1 & 2
2. [HAR-using-PyTorch](https://github.com/sidharthgurbani/HAR-using-PyTorch) - BiLSTM reference
3. [har-with-imu-transformer](https://github.com/yolish/har-with-imu-transformer) - Model 5

## 🎓 Research Paper Usage

### Comparative Analysis
This project structure supports a comprehensive comparison paper:

1. **Baseline**: Model 1 (CNN-LSTM)
2. **Attention Enhancement**: Model 2 (CNN-LSTM-Attention)
3. **Bidirectional Extension**: Model 4 (BiLSTM-Attention)
4. **SOTA Comparison**: Model 5 (Transformer)

### Metrics to Report
- Overall accuracy
- Per-class F1-scores
- Confusion matrices (available in `results/` as PNG files)
- Training time
- Model complexity (parameters)
- Inference time

## 🛠️ Development Workflow

### Completed
- ✅ Model 1 & 2 trained (93.08%, 92.64%)
- ✅ Model 4 implementation & training (93.38%)
- ✅ Model 5 configured & trained (93.48% 🏆)
- ✅ Dataset configuration
- ✅ Virtual environment setup
- ✅ Documentation
- ✅ Comparative analysis
- ✅ Research paper written

### All Tasks Complete ✅

### Dataset
- [UCI-HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

## 👥 Project Information

**Created**: November 22, 2025  
**Updated**: January 21, 2026  
**Purpose**: Comparative study for HAR research paper  
**Main Contribution**: Benchmarking 4 neural architectures for HAR

---

**Current Status**: ✅ All models trained | 🏆 Best: Transformer (93.48%) | ⭐ Best Efficiency: BiLSTM-Attention (93.38%)
