# Testing Script for Model 1: CNN-LSTM Baseline
# Loads trained model and generates confusion matrix

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION (EXACT SAME AS TRAINING)
# ============================================================================
DATASET_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/human+activity+recognition+using+smartphones/UCI HAR Dataset/"
MODEL_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/results/model1-cnn-lstm/best_model.pth"
OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters (EXACT from train_3070ti.py)
BATCH_SIZE = 64
DROPOUT = 0.1

# Activity labels
ACTIVITY_LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

# ============================================================================
# Device Setup
# ============================================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'✅ Testing on CUDA GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('✅ Testing on Apple Silicon GPU (MPS)')
else:
    device = torch.device('cpu')
    print('⚠️  Testing on CPU')

# ============================================================================
# Model Architecture (EXACT from train_3070ti.py)
# ============================================================================
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
        self.dropout = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128 * 128, 6)

    def forward(self, x):
        # x: (batch, 9, 128)
        out = self.conv1(x)      # (batch, 64, 63)
        out = self.conv2(out)    # (batch, 128, 32)
        out = self.dropout(out)
        # LSTM: input_size=32 (features), sequence=128 (channels treated as seq)
        out, _ = self.lstm(out)  # (batch, 128, 128)
        out = torch.tanh(out)
        out = out.reshape(out.size(0), -1)  # Flatten to (batch, 128*128)
        out = self.fc(out)
        return out

# ============================================================================
# Data Loading (EXACT from train_3070ti.py)
# ============================================================================
def load_inertial_signals(dataset_path, dataset_type='train'):
    signals = []
    signal_types = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    
    for signal_type in signal_types:
        filename = f'{dataset_path}{dataset_type}/Inertial Signals/{signal_type}_{dataset_type}.txt'
        signal_data = np.loadtxt(filename, dtype=np.float64)
        signals.append(signal_data)
    
    X = np.stack(signals, axis=1)  # (samples, 9, 128) - channels first for CNN
    
    y_file = f'{dataset_path}{dataset_type}/y_{dataset_type}.txt'
    y = np.loadtxt(y_file, dtype=np.int64) - 1
    
    return X, y

class UCIHARDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# Testing and Confusion Matrix Generation
# ============================================================================
def test_model():
    print("\n" + "="*60)
    print("MODEL 1: CNN-LSTM BASELINE - TESTING")
    print("="*60)
    
    # Load test data
    print("\n📊 Loading UCI-HAR test dataset...")
    X_test, y_test = load_inertial_signals(DATASET_PATH, 'test')
    print(f"   Test samples: {X_test.shape}")
    
    # Create dataset and loader
    test_dataset = UCIHARDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model and load weights
    model = CNNLSTM().to(device)
    print(f"\n📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n📂 Loading model from: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # Inference
    all_preds = []
    all_labels = []
    
    print("\n🔄 Running inference...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"\n" + "="*60)
    print(f"📊 TEST RESULTS")
    print(f"="*60)
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Macro F1 Score: {f1:.4f}")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=ACTIVITY_LABELS))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ACTIVITY_LABELS, yticklabels=ACTIVITY_LABELS)
    plt.title(f'Model 1: CNN-LSTM Baseline\nAccuracy: {accuracy:.2f}% | F1: {f1:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_PATH, 'confusion_matrix_model1_cnn_lstm.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to: {output_file}")
    plt.close()
    
    # Also save normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=ACTIVITY_LABELS, yticklabels=ACTIVITY_LABELS)
    plt.title(f'Model 1: CNN-LSTM Baseline (Normalized)\nAccuracy: {accuracy:.2f}% | F1: {f1:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_file_norm = os.path.join(OUTPUT_PATH, 'confusion_matrix_model1_cnn_lstm_normalized.png')
    plt.savefig(output_file_norm, dpi=150, bbox_inches='tight')
    print(f"✅ Normalized confusion matrix saved to: {output_file_norm}")
    plt.close()
    
    print(f"\n" + "="*60)
    print(f"✅ TESTING COMPLETE!")
    print(f"="*60)
    
    return accuracy, f1, cm

if __name__ == "__main__":
    test_model()
