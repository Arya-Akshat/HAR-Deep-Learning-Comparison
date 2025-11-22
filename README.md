# HAR Deep Learning Comparison

A comprehensive comparative study of deep learning architectures for Human Activity Recognition (HAR) using the UCI-HAR dataset. This research evaluates the performance of CNN-LSTM, attention mechanisms, BiLSTM, and Transformer models on smartphone sensor data.

## 🎯 Research Objective

Compare and benchmark different deep learning architectures to determine the most effective approach for classifying human activities from smartphone accelerometer and gyroscope data.

## 📊 Key Results

| Model | Architecture | Test Accuracy | F1-Score |
|-------|-------------|---------------|----------|
| Model 1 | CNN-LSTM Baseline | **91.11%** | **0.9113** ⭐ |
| Model 2 | CNN-LSTM-Attention | 90.94% | 0.9104 |
| Model 4 | CNN-BiLSTM-Attention | **91.21%** | 0.9029 |
| Model 5 | CNN-Transformer | 61.55% | 0.5789 ⚠️ |

**Winner:** Model 1 (CNN-LSTM Baseline) - Best balance of accuracy, F1-score, and training efficiency.

## 🔬 Study Highlights

### Main Findings

1. **Simple beats complex**: CNN-LSTM baseline (91.11%) outperformed attention-enhanced variants in efficiency and generalization
2. **Attention paradox**: Adding attention to CNN-LSTM actually decreased performance by 0.17%
3. **Transformer failure**: Modern architecture severely underperformed (61.55%) due to:
   - Data preprocessing mismatch (6 vs 9 input channels)
   - Insufficient dataset size for transformer training (7,352 samples)
   - Catastrophic failure on static postures (SITTING: 5% recall)
4. **Performance ceiling**: Traditional RNN models consistently achieved ~91% accuracy ceiling

### Technical Insights

- **LSTM/BiLSTM architectures** are superior for small-to-medium time-series datasets
- **Attention mechanisms** require careful architecture integration - not universally beneficial
- **Transformers** need proper feature engineering and larger datasets to reach potential
- **Dataset size matters**: 7K samples insufficient for transformer, adequate for LSTM

## 📁 Repository Structure

```
├── model1-cnn-lstm/              # CNN-LSTM Baseline (91.11%) ⭐
├── model2-cnn-lstm-attention/    # CNN-LSTM + Attention (90.94%)
├── model4-cnn-bilstm-attention/  # CNN-BiLSTM + Attention (91.21%)
├── model5-cnn-transformer/       # CNN-Transformer (61.55%, needs fixes)
├── bilstm-reference/             # Reference BiLSTM implementations
├── human+activity+recognition+using+smartphones/  # UCI-HAR dataset
├── RESULTS_COMPARISON.md         # Detailed analysis and comparison
├── PROJECT_STRUCTURE.md          # Complete project documentation
└── README.md                     # This file
```

Each model directory contains:
- Training notebook (Jupyter) with full pipeline
- Model architecture implementation
- Data preprocessing utilities
- Individual README with model-specific details

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PyTorch 2.x with MPS/CUDA support
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
- **Hardware**: MacBook M4 Air with MPS GPU
- **Dataset**: UCI-HAR (7,352 train, 2,947 test samples)
- **Input**: 9 features (3-axis accelerometer + 3-axis gyroscope)
- **Sequence length**: 128 timesteps
- **Classes**: 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)

## 📖 Documentation

- **[RESULTS_COMPARISON.md](RESULTS_COMPARISON.md)**: Comprehensive comparison with:
  - Detailed performance metrics
  - Per-class analysis
  - Training curves and logs
  - Architecture descriptions
  - Comparative analysis and insights
  - Future work recommendations

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Complete project organization and file descriptions

- **Individual Model READMEs**: Architecture-specific documentation in each model directory

## 🎓 Key Takeaways

### For Practitioners

- **Use CNN-LSTM (Model 1)** for production HAR systems - best accuracy/efficiency trade-off
- LSTM models converge faster (15 epochs) vs BiLSTM (90 epochs)
- Simpler architectures are easier to maintain and debug
- ~91% accuracy appears to be the ceiling for this dataset with current methods

### For Researchers

- Attention mechanisms don't universally improve performance
- Transformers require careful data preparation and sufficient training data
- Architecture-data compatibility is critical
- Small datasets (7K samples) favor LSTM over Transformer architectures

## 🔧 Troubleshooting

**Model 5 (Transformer) low accuracy?**
- Known issue: Data preprocessing loads only 6 of 9 input channels
- Fix pending: Update data loading to include all UCI-HAR features
- See RESULTS_COMPARISON.md for detailed analysis

**MPS device errors?**
- Ensure tensors are float32 (MPS doesn't support float64)
- All notebooks handle this automatically

**NumPy compatibility issues?**
- Fixed in all models: np.float → np.float64, np.int → np.int64

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
- Fix Model 5 data preprocessing
- Add data augmentation techniques
- Implement ensemble methods
- Explore other transformer variants
- Add real-time inference capabilities

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

## 📜 License

This project is open source and available under the MIT License.

---

**Last Updated**: November 22, 2025  
**Status**: Study complete - 4 models trained and analyzed
