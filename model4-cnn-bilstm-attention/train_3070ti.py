# CNN-BiLSTM-Attention Training Script for RTX 3070Ti
# Model 4: CNN-BiLSTM with Temporal Attention

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/human+activity+recognition+using+smartphones/UCI HAR Dataset/"
RESULTS_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/results/model4-cnn-bilstm-attention/"

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0015
DROPOUT = 0.5
N_HIDDEN = 128
N_LAYERS = 2

# ============================================================================
# Device Setup
# ============================================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'✅ Training on CUDA GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('✅ Training on Apple Silicon GPU (MPS)')
else:
    device = torch.device('cpu')
    print('⚠️  Training on CPU')

# ============================================================================
# Attention Mechanism
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
# Model Architecture
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
# Data Loading
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
# Training
# ============================================================================
def train_model():
    print("\n" + "="*60)
    print("MODEL 4: CNN-BiLSTM-ATTENTION")
    print("="*60)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Dropout: {DROPOUT}")
    print(f"   Hidden size: {N_HIDDEN}")
    print(f"   LSTM layers: {N_LAYERS}")
    
    # Load data
    print("\n📊 Loading UCI-HAR dataset...")
    X_train, y_train = load_inertial_signals(DATASET_PATH, 'train')
    X_test, y_test = load_inertial_signals(DATASET_PATH, 'test')
    
    print(f"   Training samples: {X_train.shape}")
    print(f"   Test samples: {X_test.shape}")
    
    # Create datasets
    train_dataset = UCIHARDataset(X_train, y_train)
    test_dataset = UCIHARDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    model = BiLSTMAttention().to(device)
    print(f"\n📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    
    # Training loop
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    best_f1 = 0.0
    
    print(f"\n🚀 Starting training...\n")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = 100 * test_correct / test_total
        test_f1 = f1_score(all_labels, all_preds, average='macro')
        test_accuracies.append(test_accuracy)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_f1 = test_f1
            torch.save(model.state_dict(), f'{RESULTS_PATH}best_model.pth')
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | F1: {test_f1:.4f}")
    
    print(f"\n" + "="*60)
    print(f"✅ TRAINING COMPLETE!")
    print(f"="*60)
    print(f"🏆 Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")
    
    # Final evaluation
    model.load_state_dict(torch.load(f'{RESULTS_PATH}best_model.pth', weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                       'SITTING', 'STANDING', 'LAYING']
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=activity_labels))
    
    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=activity_labels, yticklabels=activity_labels)
    plt.title(f'Model 4: BiLSTM-Attention Confusion Matrix\nAccuracy: {best_accuracy:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_PATH}confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model 4: BiLSTM-Attention Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Model 4: BiLSTM-Attention Training & Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_PATH}training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Results saved to: {RESULTS_PATH}")
    print(f"   - best_model.pth")
    print(f"   - confusion_matrix.png")
    print(f"   - training_curves.png")
    
    return best_accuracy, best_f1

if __name__ == "__main__":
    train_model()
