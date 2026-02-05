"""
HAR Deep Learning Demo - Single Sample Prediction
Demonstrates model predictions on selected train and test samples.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "D:/Code/AIML/HAR-Deep-Learning-Comparison/human+activity+recognition+using+smartphones/UCI HAR Dataset/"
RESULTS_PATH = "D:/Code/AIML/HAR-Deep-Learning-Comparison/results/"

# Selected samples (easy classes with high accuracy)
TRAIN_SAMPLE_IDX = 78   # WALKING sample from training set
TEST_SAMPLE_IDX = 55    # LAYING sample from test set

ACTIVITY_LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

# ============================================================================
# Device Setup
# ============================================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('✅ Using Apple Silicon GPU (MPS)')
else:
    device = torch.device('cpu')
    print('⚠️  Using CPU')

# ============================================================================
# Model Definitions (copied from training scripts for standalone use)
# ============================================================================

# Model 1: CNN-LSTM Baseline
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=64, kernel_size=6, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128 * 128, 6)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out, _ = self.lstm(out)
        out = torch.tanh(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# Model 2: CNN-LSTM with Attention
class TemporalAttn(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttn, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        score_first_part = self.fc1(hidden_states)
        h_t = hidden_states[:, -1, :]
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = nn.functional.softmax(score, dim=1)
        context_vector = torch.bmm(hidden_states.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)
        return attention_vector, attention_weights


class CNNLSTMAttention(nn.Module):
    def __init__(self):
        super(CNNLSTMAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=64, kernel_size=6, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
        self.attn = TemporalAttn(hidden_size=128)
        self.fc = nn.Linear(128, 6)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out, _ = self.lstm(out)
        out, _ = self.attn(out)
        out = self.fc(out)
        return out


# Model 4: BiLSTM with Attention
class BiLSTMAttention(nn.Module):
    def __init__(self, n_input=9, n_hidden=128, n_layers=2, n_classes=6, drop_prob=0.5):
        super(BiLSTMAttention, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        # Bidirectional LSTM with hidden//2 per direction
        self.lstm1 = nn.LSTM(n_input, n_hidden // 2, n_layers,
                             bidirectional=True, dropout=drop_prob, batch_first=True)
        # Attention mechanism (uses TemporalAttn class defined above)
        self.attn = TemporalAttn(hidden_size=n_hidden)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.attn(out)
        out = self.fc(out)
        return out


# Model 5: CNN-Transformer
class IMUTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer_dim = config.get("transformer_dim")
        self.input_proj = nn.Sequential(
            nn.Conv1d(config.get("input_dim"), self.transformer_dim, 1), nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU()
        )
        self.window_size = config.get("window_size")
        self.encode_position = config.get("encode_position")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=config.get("nhead"),
            dim_feedforward=config.get("dim_feedforward"),
            dropout=config.get("transformer_dropout"),
            activation=config.get("transformer_activation")
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.get("num_encoder_layers"),
            norm=nn.LayerNorm(self.transformer_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)
        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))
        num_classes = config.get("num_classes")
        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, self.transformer_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim // 4, num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        src = data.get('imu')
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])
        if self.encode_position:
            src += self.position_embed
        target = self.transformer_encoder(src)[0]
        target = self.log_softmax(self.imu_head(target))
        return target


# ============================================================================
# Data Loading
# ============================================================================
def load_single_sample(dataset_path, dataset_type, sample_idx):
    """Load a single sample from UCI-HAR dataset"""
    signals = []
    signal_types = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    
    for signal_type in signal_types:
        filename = f'{dataset_path}{dataset_type}/Inertial Signals/{signal_type}_{dataset_type}.txt'
        signal_data = np.loadtxt(filename, dtype=np.float64)
        signals.append(signal_data[sample_idx])
    
    # Stack signals: (128, 9) for sequence-first models
    X = np.stack(signals, axis=1)
    
    # Load label
    y_file = f'{dataset_path}{dataset_type}/y_{dataset_type}.txt'
    y = np.loadtxt(y_file, dtype=np.int64)[sample_idx] - 1
    
    return X, y


def print_sample_info(X, y, sample_type, sample_idx):
    """Print sample information"""
    print(f"\n{'='*60}")
    print(f"📊 {sample_type.upper()} SAMPLE (Index: {sample_idx})")
    print(f"{'='*60}")
    print(f"   True Label: {y} ({ACTIVITY_LABELS[y]})")
    print(f"   Shape: {X.shape} (128 timesteps x 9 sensor channels)")
    print(f"\n   Sensor Data Summary (first 5 timesteps):")
    print(f"   {'Signal':<15} {'Min':>10} {'Max':>10} {'Mean':>10}")
    print(f"   {'-'*45}")
    signal_names = ['body_acc_x', 'body_acc_y', 'body_acc_z', 
                    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                    'total_acc_x', 'total_acc_y', 'total_acc_z']
    for i, name in enumerate(signal_names):
        print(f"   {name:<15} {X[:,i].min():>10.4f} {X[:,i].max():>10.4f} {X[:,i].mean():>10.4f}")


# ============================================================================
# Model Loading and Prediction
# ============================================================================
def load_model(model_choice):
    """Load selected model with weights"""
    if model_choice == 1:
        model = CNNLSTM()
        model_path = f"{RESULTS_PATH}model1-cnn-lstm/best_model.pth"
        input_format = "channels_first"  # (batch, 9, 128)
        model_name = "CNN-LSTM Baseline"
    
    elif model_choice == 2:
        model = CNNLSTMAttention()
        model_path = f"{RESULTS_PATH}model2-cnn-lstm-attention/best_model.pth"
        input_format = "channels_first"  # (batch, 9, 128)
        model_name = "CNN-LSTM-Attention"
    
    elif model_choice == 4:
        model = BiLSTMAttention()
        model_path = f"{RESULTS_PATH}model4-cnn-bilstm-attention/best_model.pth"
        input_format = "sequence_first"  # (batch, 128, 9)
        model_name = "BiLSTM-Attention"
    
    elif model_choice == 5:
        config = {
            "input_dim": 9, "num_classes": 6, "window_size": 128,
            "transformer_dim": 256, "nhead": 8, "num_encoder_layers": 4,
            "dim_feedforward": 512, "transformer_dropout": 0.3,
            "transformer_activation": "gelu", "encode_position": True
        }
        model = IMUTransformerEncoder(config)
        model_path = f"{RESULTS_PATH}model5-cnn-transformer/best_model.pth"
        input_format = "transformer"  # dict with 'imu' key, (batch, 128, 9)
        model_name = "CNN-Transformer (Ultimate)"
    
    else:
        raise ValueError(f"Invalid model choice: {model_choice}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, input_format, model_name


def predict(model, X, input_format):
    """Run prediction on a single sample"""
    with torch.no_grad():
        if input_format == "channels_first":
            # (128, 9) -> (1, 9, 128)
            x = torch.from_numpy(X).float().T.unsqueeze(0).to(device)
            output = model(x)
        elif input_format == "sequence_first":
            # (128, 9) -> (1, 128, 9)
            x = torch.from_numpy(X).float().unsqueeze(0).to(device)
            output = model(x)
        elif input_format == "transformer":
            # (128, 9) -> dict with 'imu': (1, 128, 9)
            x = torch.from_numpy(X).float().unsqueeze(0).to(device)
            output = model({'imu': x})
        
        pred = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item() * 100
        
    return pred, confidence


# ============================================================================
# Main Demo
# ============================================================================
def main():
    print("\n" + "="*60)
    print("🎯 HAR DEEP LEARNING - SINGLE SAMPLE PREDICTION DEMO")
    print("="*60)
    
    # Load samples
    X_train, y_train = load_single_sample(DATASET_PATH, 'train', TRAIN_SAMPLE_IDX)
    X_test, y_test = load_single_sample(DATASET_PATH, 'test', TEST_SAMPLE_IDX)
    
    # Print sample information
    print_sample_info(X_train, y_train, "Training", TRAIN_SAMPLE_IDX)
    print_sample_info(X_test, y_test, "Test", TEST_SAMPLE_IDX)
    
    # Model selection
    print("\n" + "="*60)
    print("🔧 SELECT A MODEL")
    print("="*60)
    print("   1. Model 1: CNN-LSTM Baseline (93.08%)")
    print("   2. Model 2: CNN-LSTM-Attention (92.64%)")
    print("   4. Model 4: BiLSTM-Attention (93.38%)")
    print("   5. Model 5: CNN-Transformer Ultimate (93.48%)")
    print()
    
    while True:
        try:
            choice = int(input("   Enter model number (1/2/4/5): "))
            if choice in [1, 2, 4, 5]:
                break
            print("   ❌ Please enter 1, 2, 4, or 5")
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    # Load model
    print(f"\n📂 Loading Model {choice}...")
    model, input_format, model_name = load_model(choice)
    print(f"   ✅ Loaded: {model_name}")
    
    # Run predictions
    print("\n" + "="*60)
    print("🔮 PREDICTIONS")
    print("="*60)
    
    # Training sample prediction
    pred_train, conf_train = predict(model, X_train, input_format)
    train_correct = pred_train == y_train
    
    print(f"\n   📊 TRAINING Sample (Index {TRAIN_SAMPLE_IDX}):")
    print(f"      True Label:  {y_train} ({ACTIVITY_LABELS[y_train]})")
    print(f"      Prediction:  {pred_train} ({ACTIVITY_LABELS[pred_train]})")
    print(f"      Confidence:  {conf_train:.1f}%")
    print(f"      Result:      {'✅ CORRECT' if train_correct else '❌ WRONG'}")
    
    # Test sample prediction
    pred_test, conf_test = predict(model, X_test, input_format)
    test_correct = pred_test == y_test
    
    print(f"\n   📊 TEST Sample (Index {TEST_SAMPLE_IDX}):")
    print(f"      True Label:  {y_test} ({ACTIVITY_LABELS[y_test]})")
    print(f"      Prediction:  {pred_test} ({ACTIVITY_LABELS[pred_test]})")
    print(f"      Confidence:  {conf_test:.1f}%")
    print(f"      Result:      {'✅ CORRECT' if test_correct else '❌ WRONG'}")
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print(f"   Model: {model_name}")
    print(f"   Training Sample: {'✅ CORRECT' if train_correct else '❌ WRONG'}")
    print(f"   Test Sample:     {'✅ CORRECT' if test_correct else '❌ WRONG'}")
    
    if train_correct and test_correct:
        print("\n   🎉 Both predictions are correct!")
    else:
        print("\n   ⚠️  Some predictions were wrong. Try different samples.")
    
    print("="*60 + "\n")
    
    return train_correct, test_correct


if __name__ == "__main__":
    main()
