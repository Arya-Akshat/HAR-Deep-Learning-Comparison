# CNN-LSTM Training Script for RTX 3070Ti
# Model 1: Baseline CNN-LSTM

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/human+activity+recognition+using+smartphones/UCI HAR Dataset/"
RESULTS_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/results/model1-cnn-lstm/"

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT = 0.1

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
# Model Architecture
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
# Training
# ============================================================================
def train_model():
    print("\n" + "="*60)
    print("MODEL 1: CNN-LSTM BASELINE")
    print("="*60)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Dropout: {DROPOUT}")
    
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
    model = CNNLSTM().to(device)
    print(f"\n📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_f1 = test_f1
            torch.save(model.state_dict(), f'{RESULTS_PATH}best_model.pth')
        
        if epoch % 5 == 0 or epoch == 1:
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_labels, yticklabels=activity_labels)
    plt.title(f'Model 1: CNN-LSTM Confusion Matrix\nAccuracy: {best_accuracy:.2f}%')
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
    ax1.set_title('Model 1: CNN-LSTM Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Model 1: CNN-LSTM Training & Test Accuracy')
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
