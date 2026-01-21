# HAR Deep Learning Comparison

A comprehensive comparative study of deep learning architectures for Human Activity Recognition (HAR) using the UCI-HAR dataset. This research evaluates the performance of CNN-LSTM, attention mechanisms, BiLSTM, and Transformer models on smartphone sensor data.

## 🎯 Research Objective

Compare and benchmark different deep learning architectures to determine the most effective approach for classifying human activities from smartphone accelerometer and gyroscope data.

## 📊 Key Results (Updated December 2025)

| Rank | Model | Architecture | Test Accuracy | F1-Score | Parameters |
|------|-------|-------------|---------------|----------|------------|
| 🥇 | Model 5 | CNN-Transformer (Ultimate) | **93.48%** | **0.9341** | 2,359,494 |
| 🥈 | Model 4 | CNN-BiLSTM-Attention | 93.38% | 0.9350 | 187,654 |
| 🥉 | Model 1 | CNN-LSTM Baseline | 93.08% | 0.9309 | 209,478 |
| 4th | Model 2 | CNN-LSTM-Attention | 92.64% | 0.9266 | 161,094 |

**Winner:** Model 5 (CNN-Transformer Ultimate) - Achieves best accuracy with optimized architecture and training.

## 🔬 Study Highlights

### Main Findings

1. **Transformers can win** with proper optimization: Larger model (256 dim, 4 layers) + longer training (150 epochs) + warmup scheduler achieved **93.48%**
2. **Attention paradox**: Adding attention to CNN-LSTM actually **decreased** performance by 0.44%
3. **BiLSTM strength**: Bidirectional context helps - nearly matched Transformer with 12x fewer parameters
4. **SITTING vs STANDING**: Hardest classification pair across ALL models (75-85% recall)

### Technical Insights

- **Transformer optimization** is critical: warmup scheduler, gradient clipping, label smoothing
- **Attention mechanisms** don't universally improve CNN-LSTM architectures
- **BiLSTM** offers best accuracy-to-parameters ratio
- **Dataset size**: 7K samples works for all architectures with proper training

## 📁 Repository Structure

```
├── model1-cnn-lstm/              # CNN-LSTM Baseline (93.08%) 🥉
├── model2-cnn-lstm-attention/    # CNN-LSTM + Attention (92.64%)
├── model4-cnn-bilstm-attention/  # CNN-BiLSTM + Attention (93.38%) 🥈
├── model5-cnn-transformer/       # CNN-Transformer Ultimate (93.48%) 🥇
├── bilstm-reference/             # Reference BiLSTM implementations
├── human+activity+recognition+using+smartphones/  # UCI-HAR dataset
├── results/                      # Training outputs (checkpoints, log)
├── RESULTS_COMPARISON.md         # Detailed analysis and comparison
├── PROJECT_STRUCTURE.md          # Complete project documentation
└── README.md                     # This file
```

Each model directory contains:
- `train_3070ti.py` - Optimized training script (Windows/CUDA)
- Training notebook (Jupyter) with full pipeline
- Model architecture implementation
- Data preprocessing utilities
- Individual README with model-specific details

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PyTorch 2.x with CUDA support
- UV package manager (recommended) or pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Arya-Akshat/HAR-Deep-Learning-Comparison.git
cd HAR-Deep-Learning-Comparison
```

2. **Download UCI-HAR dataset**

Download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) and extract to `human+activity+recognition+using+smartphones/`

3. **Setup virtual environment**

Each model uses UV for fast dependency management:
```bash
cd model1-cnn-lstm  # or any model directory
# Run the first cell in the Jupyter notebook to setup venv
```

4. **Train models**

Open any `Train-*.ipynb` notebook and run all cells. Each notebook includes:
- Automatic virtual environment setup
- Dataset loading and preprocessing
- Model training with validation
- Results visualization
- Model checkpoints saving

## 📈 Training Configuration

All models trained on:
- **Hardware**: NVIDIA GeForce RTX 3070 Ti Laptop GPU (8.6GB VRAM)
- **Dataset**: UCI-HAR (7,352 train, 2,947 test samples)
- **Input**: 9 channels (body_acc xyz, body_gyro xyz, total_acc xyz)
- **Sequence length**: 128 timesteps
- **Classes**: 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)

### Optimized Training Scripts

Each model has a `train_3070ti.py` script with:
- Windows-compatible paths
- Automatic GPU detection (CUDA/MPS/CPU)
- Results saved to `results/modelX/` folder
- Summary metrics tracked in `results/TRAINING_LOG.md`

```bash
# Run any model
python model1-cnn-lstm/train_3070ti.py
python model2-cnn-lstm-attention/train_3070ti.py
python model4-cnn-bilstm-attention/train_3070ti.py
python model5-cnn-transformer/train_ultimate.py  # Best results
```

## 📖 Documentation

- **[RESULTS_COMPARISON.md](RESULTS_COMPARISON.md)**: Comprehensive comparison with:
  - Detailed performance metrics
  - Per-class analysis
  - Training log summary (authoritative results in `results/TRAINING_LOG.md`)
  - Architecture descriptions
  - Comparative analysis and insights
  - Future work recommendations

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Complete project organization and file descriptions

- **Individual Model READMEs**: Architecture-specific documentation in each model directory

## 🎓 Key Takeaways

### For Practitioners

- **Use CNN-Transformer (Model 5)** for best accuracy if compute budget allows
- **Use BiLSTM (Model 4)** for best accuracy/efficiency trade-off
- SITTING vs STANDING is inherently difficult (~85% recall max)
- All models achieve 93%+ with proper optimization

### For Researchers

- Attention mechanisms don't universally improve performance
- Transformers CAN work on small datasets with proper optimization
- Warmup schedulers are critical for transformer training
- Architecture-data compatibility matters less than training strategy

## 🔧 Troubleshooting

**Dataset path errors?**
- Update `DATASET_PATH` in `train_3070ti.py` to match your local path

**CUDA out of memory?**
- Reduce batch size in the training script

**MPS device errors?**
- Ensure tensors are float32 (MPS doesn't support float64)

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{har_comparison_2025,
  author = {Arya-Akshat},
  title = {HAR Deep Learning Comparison: A Comparative Study of Deep Learning Architectures for Human Activity Recognition},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Arya-Akshat/HAR-Deep-Learning-Comparison}
}
```

## 📄 Dataset Citation

```bibtex
@article{anguita2013,
  title={A public domain dataset for human activity recognition using smartphones},
  author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge L},
  journal={Esann},
  year={2013}
}
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add data augmentation techniques
- Implement ensemble methods
- Explore other transformer variants (PatchTST, Informer)
- Add real-time inference capabilities
- Cross-dataset validation

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

## 📜 License

This project is open source and available under the MIT License.

---

**Last Updated**: December 20, 2025  
**Status**: Study complete - 4 models trained and benchmarked (93.48% best accuracy)

> **Source of truth**: Only the outputs under `results/` are authoritative. Use `results/TRAINING_LOG.md` for the latest metrics and the `best_model.pth` checkpoints for each model.
