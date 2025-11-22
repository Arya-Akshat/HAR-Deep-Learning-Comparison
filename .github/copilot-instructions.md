# Human Activity Recognition (HAR) Projects - AI Agent Instructions

## Research Context & Paper Implementation Goals

This workspace is for a **comparative study paper** using PyTorch on the **UCI-HAR dataset**. The goal is to compare four architectures with minimal implementation work by leveraging existing repositories.

### Paper Models & Repository Mapping

1. **CNN-LSTM (Baseline)** → `LizLicense/HAR-CNN-LSTM-ATT-pyTorch` ✅ Already implemented
2. **CNN-LSTM-Attention (Proposed)** → `LizLicense/HAR-CNN-LSTM-ATT-pyTorch` ✅ Already implemented
3. **CNN-BiLSTM-Attention** → `sidharthgurbani/HAR-using-PyTorch` + Attention layer from repo #1
4. **CNN-Transformer (SOTA)** → `yolish/har-with-imu-transformer` (configured for UCI-HAR)

**Implementation Strategy**: Repository #1 contains both baseline and proposed models. Minimal work needed: add attention layer to BiLSTM model from repo #2, and adapt transformer from repo #3.

## Project Overview
This workspace contains three complementary HAR research projects using PyTorch, each implementing different deep learning architectures for time-series sensor data classification.

### Architecture Map
1. **HAR-CNN-LSTM-ATT-pyTorch** (`LizLicense`): Main project with CNN-LSTM (baseline) + CNN-LSTM-Attention (proposed) + self-supervised learning (SSL)
2. **HAR-using-PyTorch** (`sidharthgurbani`): LSTM variants including BiLSTM (base for Model 3)
3. **har-with-imu-transformer** (`yolish`): Transformer-based approach for IMU data (Model 4)

## Critical Development Patterns

### Dataset Structure & Paths
- **Primary Dataset**: **UCI-HAR** (9 channels, 6 activity classes) - used for paper comparisons
  - Additional datasets: HAPT (6 channels, 12 classes), HHAR (3 channels, phone/watch variants)
  - UCI: 9 channels (9D sensor data from accelerometer + gyroscope)
  - HAPT: 6 channels with 12 activity classes
  - HHAR: 3 channels (accelerometer/gyroscope split: phone vs watch)
- Data stored as `.pt` files (PyTorch tensors) in `{uci,hapt,hhar}_data/` folders
- Training data naming: `train_{percentage}per.pt` (e.g., `train_100per.pt`, `train_10per.pt`)
- Expected tensor shape: `(batch, channels, sequence_length)` - always ensure channels in second dimension
- **UCI-HAR Dataset Sources**: 
  - Raw data: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/
  - Already preprocessed in `uci_data/` folder as `.pt` files

### Training Modes (HAR-CNN-LSTM-ATT-pyTorch)
Three distinct modes controlled via `--training_mode` argument:
- `"ssl"`: Self-supervised learning with augmentation-based pretraining
- `"ft"`: Fine-tuning from SSL checkpoint (loads `ssl_checkpoint.pt`)
- `"supervised"`: Standard supervised training

**Key Pattern**: Fine-tuning requires SSL checkpoint to exist first. Shell scripts (e.g., `uci_run_uci.sh`) run SSL → FT → Supervised sequentially.

### Device Configuration
Device selection follows this priority pattern:
```python
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
```
Or via argument: `--device cpu|mps|cuda:0`

### Network Architecture Conventions

#### CNN-LSTM Pattern (supervised/)
- Conv1D → ReLU → MaxPool → Conv1D → ReLU → MaxPool → Dropout → LSTM → Flatten → FC
- Without attention: flatten LSTM output directly (shape: `batch × (hidden_size × seq_len)`)
- With attention: `Attention/` subfolder, attention layer before flatten (shape: `batch × hidden_size`)
- **Critical**: Attention changes FC input dimensions from `128*128` to `128`

#### SSL Architecture (ssl/)
- Separate components: `backbone_fe` (CNN feature extractor) + `backbone_temporal` (identity) + `classifier`
- `ssl_classifier` predicts transformation type (num_tasks = len(augmentations))
- Regular `classifier` predicts activity class
- `predict_features` argument varies by dataset/architecture (e.g., 6144 for HAPT, 4352 for UCI/HHAR)

#### Config-based Architecture (HAR-using-PyTorch, har-with-imu-transformer)
- Architecture selection via `config.py` dict (e.g., `arch = LSTM1`, `Res_LSTM`)
- Model initialized based on `arch['name']` key
- Transformer uses `config.json` for all hyperparameters

### Data Augmentation (SSL)
Augmentations combined with underscores in string format:
- Example: `"permute_timeShift_scale_noise"` or `"negate_permute_timeShift_scale_noise"`
- Applied in `augmentations.py`: `jitter`, `scaling`, `permutation`, `time_shift`, `vat_noise`
- Each augmentation becomes a classification task for SSL pretraining

### Class Imbalance Handling
- Oversampling implemented via `get_balance_class_oversample()` in `data_loader.py`
- Controlled by `--oversample True|False` argument
- **Never apply to fine-tuning data** (check: `if oversample and "ft" not in training_mode`)

### Result Persistence Pattern
Results saved in parallel CSV files:
- `result_cnn-lstm_{DATASET}.csv`: Train/test accuracy per epoch
- `result_f1_{DATASET}.csv`: F1 scores per epoch
- Confusion matrices as PNG images
- Loss/accuracy plots
- Convention: Separate result folders per dataset (UCI/, HAPT/, HHAR/)

### Data Preprocessing Workflow
1. Raw data → `train_100per.pt` (full dataset)
2. Run `create_few_percetages.py` to generate subset files (1%, 5%, 10%, 50%, 75%)
3. Move `.pt` files to appropriate `{dataset}_data/` folders
4. Shell scripts iterate over percentages for experiments

### Conda Environment
- Environment name: `base` (activate via `conda activate base`)
- Key dependencies in `environment.yml`: PyTorch, imbalanced-learn, pandas, scikit-learn
- MacOS ARM64-specific setup (Miniconda3-latest-MacOSX-arm64.sh)

## Common Gotchas

1. **Checkpoint Loading**: Fine-tuning expects checkpoint at `{save_path}/{dataset}/{data_percentage}/ssl_checkpoint.pt`
2. **Permutation**: Data needs permutation `x.permute(0,2,1)` or `x.permute(1,0,2)` depending on context - check existing code
3. **Softmax Position**: Applied in model forward pass, not loss (using `nn.CrossEntropyLoss` which expects raw logits but models output softmax)
4. **Result Path**: Hardcoded relative paths like `'result/UCI/CNNLSTM/'` - create directories before running
5. **Data Folder Argument**: Must match dataset type (`../uci_data/`, `../hapt_data/`, `../hhar_data/`)

## Running Experiments

### SSL Training (HAR-CNN-LSTM-ATT-pyTorch)
```bash
cd GitFYP_experiment/ssl/
python main.py --training_mode ssl --dataset UCI --data_percentage 10 --nepoch 50
```

### Supervised Training with Attention
```bash
cd GitFYP_experiment/supervised/UCI/Attention/
python main_pytorch.py --nepoch 50 --batchsize 64
```

### Transformer Training
```bash
cd har-with-imu-transformer/
python main.py train <path_to_csv> --experiment "description"
```

## Architecture-Specific Notes

### BiLSTM (HAR-using-PyTorch) - Base for Model 3
- **Current State**: Implements `Bidir_LSTMModel` with bidirectional LSTM layers
- **Paper Task**: Add attention mechanism from `HAR-CNN-LSTM-ATT-pyTorch/GitFYP_experiment/supervised/UCI/Attention/attention.py`
- Uses custom residual blocks with `add_residual_component()` in Residual variants
- Batch normalization applied after residual addition
- Highway layers controlled by `n_highway_layers` config parameter
- **Key Files**: 
  - `LSTM/model.py`: Contains `Bidir_LSTMModel` class
  - `LSTM/config.py`: Architecture configurations (`Bidir_LSTM1`, `Bidir_LSTM2`)

### Transformer (har-with-imu-transformer) - Model 4 SOTA
- CLS token prepended to sequence
- Position embeddings optional via `encode_position` flag
- Window-based processing: CSV → windows via `IMUDataset` class
- Entry point: `main.py train|test <dataset_file>`
- **Paper Task**: Configure for UCI-HAR dataset (6 input dimensions, 6 classes in `config.json`)
- Input projection: Conv1D layers before transformer encoder

### Attention Mechanism (for Model 3 Integration)
- **Source**: `HAR-CNN-LSTM-ATT-pyTorch/GitFYP_experiment/supervised/UCI/Attention/attention.py`
- Class: `TemporalAttn(hidden_size)`
- Applied after LSTM/BiLSTM, before flatten layer
- Returns: `(attended_output, attention_weights)` where attended_output shape is `(batch, hidden_size)`
- **Critical**: Changes FC layer input from `hidden_size * seq_len` to just `hidden_size`

## File Organization Logic
- Each dataset has dedicated folder in `supervised/` with own `main_pytorch.py`, `network.py`, `data_preprocess.py`
- Attention variants live in `Attention/` subfolders with modified network architecture
- SSL code centralized in `ssl/` with dataset-specific shell scripts

## Paper Implementation Roadmap

### ✅ Already Complete
- Model 1: CNN-LSTM baseline (`HAR-CNN-LSTM-ATT-pyTorch/supervised/UCI/main_pytorch.py`)
- Model 2: CNN-LSTM-Attention (`HAR-CNN-LSTM-ATT-pyTorch/supervised/UCI/Attention/main_pytorch.py`)

### 🔨 Minimal Work Required
- **Model 3: CNN-BiLSTM-Attention**
  1. Copy `TemporalAttn` class from repo #1 to repo #2
  2. Modify `Bidir_LSTMModel` in `HAR-using-PyTorch/LSTM/model.py`
  3. Add attention layer after BiLSTM, update FC input dimensions
  4. Test on UCI-HAR dataset

- **Model 4: CNN-Transformer**
  1. Update `config.json` in `har-with-imu-transformer`: `input_dim: 6`, `num_classes: 6`
  2. Prepare UCI-HAR data in CSV format (IMUDataset expects CSV)
  3. Run training: `python main.py train <uci_har.csv>`
  4. Compare results with Models 1-3
