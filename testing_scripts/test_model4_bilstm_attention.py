# Testing Script for Model 4: CNN-BiLSTM-Attention
# Loads trained model and generates confusion matrix

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION (EXACT SAME AS TRAINING)
# ============================================================================
DATASET_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/human+activity+recognition+using+smartphones/UCI HAR Dataset/"
MODEL_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/results/model4-cnn-bilstm-attention/best_model.pth"
OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters (EXACT from train_3070ti.py)
BATCH_SIZE = 64
DROPOUT = 0.5
N_HIDDEN = 128
N_LAYERS = 2

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
# Attention Mechanism (EXACT from train_3070ti.py)
# ============================================================================
class TemporalAttn(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttn, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, time_steps, hidden_size)
        score_first_part = self.fc1(hidden_states)
        h_t = hidden_states[:, -1, :]
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)
        context_vector = torch.bmm(hidden_states.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)
        return attention_vector, attention_weights

# ============================================================================
# Model Architecture (EXACT from train_3070ti.py)
# ============================================================================
class BiLSTMAttention(nn.Module):
    def __init__(self, n_input=9, n_hidden=N_HIDDEN, n_layers=N_LAYERS, n_classes=6, drop_prob=DROPOUT):
        super(BiLSTMAttention, self).__init__()
        
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        
        # Bidirectional LSTM
        self.lstm1 = nn.LSTM(n_input, n_hidden // 2, n_layers, 
                             bidirectional=True, dropout=drop_prob, batch_first=True)
        
        # Attention mechanism
        self.attn = TemporalAttn(hidden_size=n_hidden)
        
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        # x: (batch, 128, 9) - sequence first
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        
        # Apply attention
        out, attn_weights = self.attn(out)
        out = self.fc(out)
        return out

# ============================================================================
# Data Loading (EXACT from train_3070ti.py)
# Note: Model 4 uses (samples, 128, 9) - sequence first for BiLSTM
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
    
    # (samples, 128, 9) - sequence first for BiLSTM
    X = np.stack(signals, axis=2)
    
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
    print("MODEL 4: CNN-BiLSTM-ATTENTION - TESTING")
    print("="*60)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Dropout: {DROPOUT}")
    print(f"   Hidden size: {N_HIDDEN}")
    print(f"   LSTM layers: {N_LAYERS}")
    
    # Load test data
    print("\n📊 Loading UCI-HAR test dataset...")
    X_test, y_test = load_inertial_signals(DATASET_PATH, 'test')
    print(f"   Test samples: {X_test.shape}")
    
    # Create dataset and loader
    test_dataset = UCIHARDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model and load weights
    model = BiLSTMAttention().to(device)
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=ACTIVITY_LABELS, yticklabels=ACTIVITY_LABELS)
    plt.title(f'Model 4: CNN-BiLSTM-Attention\nAccuracy: {accuracy:.2f}% | F1: {f1:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_PATH, 'confusion_matrix_model4_bilstm_attention.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to: {output_file}")
    plt.close()
    
    # Also save normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Oranges', 
                xticklabels=ACTIVITY_LABELS, yticklabels=ACTIVITY_LABELS)
    plt.title(f'Model 4: CNN-BiLSTM-Attention (Normalized)\nAccuracy: {accuracy:.2f}% | F1: {f1:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_file_norm = os.path.join(OUTPUT_PATH, 'confusion_matrix_model4_bilstm_attention_normalized.png')
    plt.savefig(output_file_norm, dpi=150, bbox_inches='tight')
    print(f"✅ Normalized confusion matrix saved to: {output_file_norm}")
    plt.close()
    
    print(f"\n" + "="*60)
    print(f"✅ TESTING COMPLETE!")
    print(f"="*60)
    
    return accuracy, f1, cm

if __name__ == "__main__":
    test_model()
