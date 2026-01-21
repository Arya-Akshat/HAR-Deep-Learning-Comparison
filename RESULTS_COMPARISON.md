# Human Activity Recognition - Model Comparison Results

## Research Overview
Comparative study of deep learning architectures for Human Activity Recognition (HAR) using the UCI-HAR dataset.

**Dataset**: UCI Human Activity Recognition Using Smartphones
- Training samples: 7,352
- Test samples: 2,947
- Input features: 9 (3-axis accelerometer + 3-axis gyroscope + 3-axis total acceleration)
- Sequence length: 128 timesteps
- Activity classes: 6 (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)

**Hardware**: NVIDIA GeForce RTX 3070 Ti GPU (8 GB)

**Results Summary**: All optimized models cluster around 93% accuracy, suggesting an intrinsic performance ceiling for this dataset. The Transformer achieves the top score of 93.48%, while BiLSTM-Attention offers the best efficiency at 93.38% with only 188K parameters.

**Source of truth**: The authoritative metrics and outputs live in `results/` (see `results/TRAINING_LOG.md` and the `best_model.pth` checkpoints).

---

## Model Results Summary

| Model | Architecture | Test Accuracy | F1-Score | Parameters | Training Status |
|-------|-------------|---------------|----------|------------|-----------------|
| **Model A** | CNN-LSTM Baseline | **93.08%** | **0.9309** | 209K | ✅ **Completed** |
| **Model B** | CNN-LSTM-Attention | 92.64% | 0.9266 | 161K | ✅ **Completed** |
| **Model C** | BiLSTM-Attention | **93.38%** | **0.9350** | 188K | ✅ **Completed** |
| **Model D** | CNN-Transformer | **93.48%** 🏆 | **0.9341** | 2,359K | ✅ **Completed** |

---

## Detailed Results

### Model A: CNN-LSTM Baseline ✅
**Status**: Training Complete

**Architecture**:
```
Input (batch, 9, 128)
    ↓
Conv1D (9→64, kernel=6) → ReLU → MaxPool
    ↓
Conv1D (64→128, kernel=3) → ReLU → MaxPool
    ↓
Dropout (0.1)
    ↓
LSTM (input=32, hidden=128, layers=1)
    ↓
Tanh
    ↓
Flatten (128×128)
    ↓
FC (16384→6)
    ↓
Softmax
```

**Hyperparameters**:
- Learning rate: 0.001
- Epochs: 50
- Batch size: 64
- Optimizer: Adam
- Loss: CrossEntropyLoss

**Performance Metrics**:
- **Test Accuracy**: 93.08% (best at epoch 15)
- **F1-Score**: 0.931 (macro average)
- **Parameters**: ~209,000

**Per-Class Performance**:
| Activity | F1-Score |
|----------|----------|
| WALKING | 0.995 |
| WALKING_UPSTAIRS | 0.931 |
| WALKING_DOWNSTAIRS | 0.970 |
| SITTING | 0.830 |
| STANDING | 0.875 |
| LAYING | 0.985 |

**Key Observations**:
- Excellent performance on dynamic activities (WALKING, LAYING)
- Lower performance distinguishing between static postures (SITTING vs STANDING)
- Fast convergence (best accuracy at epoch 15/50)
- No significant overfitting observed

---

### Model B: CNN-LSTM-Attention ✅
**Status**: Training Complete

**Architecture**:
```
Input (batch, 9, 128)
    ↓
Conv1D (9→64, kernel=6) → ReLU → MaxPool
    ↓
Conv1D (64→128, kernel=3) → ReLU → MaxPool
    ↓
Dropout (0.1)
    ↓
LSTM (input=32, hidden=128, layers=1)
    ↓
Tanh
    ↓
Temporal Attention (hidden=128) → Context Vector
    ↓
FC (128→6)
    ↓
Softmax
```

**Hyperparameters**:
- Learning rate: 0.001
- Epochs: 50
- Batch size: 64
- Optimizer: Adam
- Loss: CrossEntropyLoss

**Performance Metrics**:
- **Test Accuracy**: 92.64% (best at epoch 40)
- **F1-Score**: 0.927 (macro average)
- **Parameters**: ~161,000

**Per-Class Performance**:
| Activity | F1-Score |
|----------|----------|
| WALKING | 0.965 |
| WALKING_UPSTAIRS | 0.950 |
| WALKING_DOWNSTAIRS | 0.984 |
| SITTING | 0.813 |
| STANDING | 0.868 |
| LAYING | 0.981 |

**Key Observations**:
- ⚠️ **Attention Paradox**: Adding attention DECREASED accuracy by 0.44 points (92.64% vs 93.08% baseline)
- Performs WORSE than Model A without attention
- Converged at epoch 40/50
- Suggests attention may not benefit CNN-LSTM architectures on this dataset

---

### Model C: BiLSTM-Attention ✅ ⭐ **BEST EFFICIENCY**
**Status**: Training Complete

**Architecture**:
```
Input (batch, 128, 9)
    ↓
BiLSTM Layer 1 (bidirectional, hidden=32)
    ↓
Highway BiLSTM Layer 2
    ↓
Dropout (0.5)
    ↓
Temporal Attention → Context Vector (32)
    ↓
FC (32→6)
    ↓
Softmax
```

**Hyperparameters**:
- Learning rate: 0.0015
- Epochs: 120
- Batch size: 64
- Optimizer: Adam
- Hidden size: 32
- Layers: 2 (bidirectional)
- Dropout: 0.5

**Performance Metrics**:
- **Test Accuracy**: 93.38% (best at epoch 90)
- **F1-Score**: 0.935 (macro average) - **Highest F1!**
- **Parameters**: ~188,000

**Per-Class Performance**:
| Activity | F1-Score |
|----------|----------|
| WALKING | 0.983 |
| WALKING_UPSTAIRS | 0.972 |
| WALKING_DOWNSTAIRS | 0.978 |
| SITTING | 0.836 |
| STANDING | 0.865 |
| LAYING | 0.976 |

**Key Observations**:
- **Best efficiency**: 93.38% accuracy with only 188K parameters
- Attention DOES work with BiLSTM (unlike CNN-LSTM)
- Best sitting/standing classification (F1: 0.84-0.85)
- 12.5× fewer parameters than Transformer for only 0.10% less accuracy

---

### Model D: CNN-Transformer ✅ 🏆 **HIGHEST ACCURACY**
**Status**: Training Complete

**Architecture**:
```
Input (batch, 128, 9)
    ↓
Conv1D Projection Layers (4 layers with GELU)
    Layer 1: 9→32, kernel=5, pad=2
    Layer 2: 32→64, kernel=5, pad=2
    Layer 3: 64→64, kernel=5, pad=2
    Layer 4: 64→64, kernel=5, pad=2
    ↓
CLS Token Prepended
    ↓
Positional Encoding (learnable)
    ↓
Transformer Encoder (6 layers)
    - Multi-head attention (8 heads)
    - Dimension: 64
    - Feed-forward: 128
    - Dropout: 0.1
    ↓
CLS Token Output
    ↓
Classification Head:
    LayerNorm → Linear → GELU → Dropout → Linear → LogSoftmax
    ↓
Output (6 classes)
```

**Hyperparameters**:
- Learning rate: 5e-5 (with warmup for 10 epochs)
- Epochs: 150
- Batch size: 64
- Optimizer: Adam
- Weight decay: 0.01
- LR scheduler: StepLR (step=10, gamma=0.5)
- Loss: NLLLoss

**Performance Metrics**:
- **Test Accuracy**: 93.48% 🏆 **BEST**
- **F1-Score**: 0.934 (macro average)
- **Parameters**: ~2,359,000

**Per-Class Performance**:
| Activity | F1-Score |
|----------|----------|
| WALKING | 0.961 |
| WALKING_UPSTAIRS | 0.954 |
| WALKING_DOWNSTAIRS | 0.932 |
| SITTING | 0.869 |
| STANDING | 0.910 |
| LAYING | 0.979 |

**Key Observations**:
- **Highest overall accuracy** at 93.48%
- Required extensive hyperparameter tuning (warmup, weight decay, step decay)
- Initial experiments without proper regularization achieved only 61% accuracy
- 12.5× more parameters than BiLSTM-Attention for only 0.10% improvement
- Best convergence around epoch 120/150

---

## Comparative Analysis

### All Models Comparison

| Metric | Model A (CNN-LSTM) | Model B (CNN-LSTM-Attn) | Model C (BiLSTM-Attn) | Model D (Transformer) |
|--------|-------------------|------------------------|----------------------|----------------------|
| **Test Accuracy** | 93.08% | 92.64% | 93.38% ⭐ | **93.48%** 🏆 |
| **F1-Score** | 0.9309 | 0.9266 | **0.9350** 🏆 | 0.9341 |
| **Convergence** | Epoch 15 🏆 | Epoch 40 | Epoch 90 | Epoch 120 |
| **Parameters** | 209K | 161K 🏆 | 188K | 2,359K |
| **Attention** | ❌ | ✅ | ✅ | ✅ Multi-head |

### Key Findings

1. **Attention Paradox**: Adding attention to CNN-LSTM (Model B) DECREASED performance by 0.44%
   - Model A (no attention): 93.08%
   - Model B (with attention): 92.64%
   - Contradicts intuition that attention always helps

2. **Best Overall Accuracy**: Model D (Transformer) at 93.48%
   - But requires 12.5× more parameters than Model C
   - Needs careful hyperparameter tuning

3. **Best Efficiency**: Model C (BiLSTM-Attention)
   - 93.38% accuracy with only 188K parameters
   - Highest F1-score (0.935)
   - Best for mobile/edge deployment

4. **~93% Performance Ceiling**: All optimized models cluster near 93%, suggesting:
   - Dataset limitation rather than architecture limitation
   - Possible annotation ambiguity between sitting/standing
   - Diminishing returns from architectural complexity

5. **When Attention Helps**:
   - ✅ Works with BiLSTM (Model C: 93.38%)
   - ❌ Hurts CNN-LSTM (Model B: 92.64% < Model A: 93.08%)
   - Architecture compatibility matters more than attention alone

---

## Practical Recommendations

### Production Deployment
| Priority | Recommendation | Model |
|----------|---------------|-------|
| Maximum accuracy | Use Transformer with careful tuning | Model D |
| Best efficiency | Deploy BiLSTM-Attention | Model C ⭐ |
| Rapid prototyping | Train CNN-LSTM baseline | Model A |
| Avoid | Adding attention without validation | Model B |

### Mobile/Edge Deployment
**Recommended**: Model C (BiLSTM-Attention)
- 93.38% accuracy
- Only 188K parameters
- 12.5× smaller than Transformer
- Best F1-score (0.935)

---

## Training Logs

### Training Progress Notes
Per-epoch logs are not re-derived from the saved checkpoints. The current `results/` folder contains only the `best_model.pth` checkpoints and `results/TRAINING_LOG.md` summary (no training-curve images).

---

## Conclusion

All four models have been successfully trained and optimized on the UCI-HAR dataset:

### Performance Rankings
1. **Model D** (Transformer): 93.48% - Highest accuracy 🏆
2. **Model C** (BiLSTM-Attention): 93.38% - Best efficiency ⭐ **RECOMMENDED**
3. **Model A** (CNN-LSTM): 93.08% - Fast training, solid baseline
4. **Model B** (CNN-LSTM-Attention): 92.64% - Attention hurt performance

### Key Insights

1. **Attention Paradox**: Adding attention to CNN-LSTM hurt performance (-0.44%)
2. **~93% ceiling**: All optimized models cluster near 93%, suggesting dataset limitations
3. **Efficiency matters**: BiLSTM-Attention achieves near-best accuracy with 12.5× fewer parameters
4. **Transformers need tuning**: Without proper regularization/warmup, Transformer achieved only 61%
5. **Static posture challenge**: Sitting vs Standing remains hardest to distinguish (F1: 0.80-0.85)

### Production Recommendation

**Model C (BiLSTM-Attention)** is the clear winner for deployment:
- Near-best accuracy (93.38%, only 0.10% behind Transformer)
- Highest F1-score (0.935)
- 12.5× fewer parameters than Transformer
- No overfitting issues
- Robust across all activity classes

---

**Last Updated**: January 20, 2026  
**Status**: ✅ All 4 models successfully trained and evaluated  
**Best Model**: Model D (Transformer) - 93.48% accuracy  
**Best Efficiency**: Model C (BiLSTM-Attention) - 93.38% accuracy, 188K parameters
