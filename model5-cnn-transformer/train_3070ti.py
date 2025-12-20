# CNN-Transformer Optimized Training for RTX 3070Ti
# Run this script to train with optimized hyperparameters

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.IMUTransformerEncoder import IMUTransformerEncoder

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/human+activity+recognition+using+smartphones/UCI HAR Dataset/"
RESULTS_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/results/model5-cnn-transformer/"
CONFIG_FILE = "config_3070ti_optimized.json"

# ============================================================================
# Device Setup
# ============================================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'✅ Training on CUDA GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('✅ Training on Apple Silicon GPU (MPS)')
else:
    device = torch.device('cpu')
    print('⚠️  Training on CPU')

# ============================================================================
# Data Loading
# ============================================================================
def load_inertial_signals(dataset_path, dataset_type='train'):
    """Load inertial signals from UCI-HAR dataset"""
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
    
    # Stack signals: (num_samples, 128, 9)
    X = np.stack(signals, axis=2)
    
    # Load labels (1-indexed, need to convert to 0-indexed)
    y_file = f'{dataset_path}{dataset_type}/y_{dataset_type}.txt'
    y = np.loadtxt(y_file, dtype=np.int64) - 1
    
    return X, y

class UCIHARDataset(Dataset):
    """UCI-HAR Dataset for Transformer model"""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'imu': self.X[idx], 'label': self.y[idx]}

# ============================================================================
# Warmup Scheduler (Critical for Transformers!)
# ============================================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = max(self.base_lrs[i] * lr_scale, self.min_lr)
        
        return self.optimizer.param_groups[0]['lr']

# ============================================================================
# Training Function
# ============================================================================
def train_model():
    print("\n" + "="*60)
    print("CNN-TRANSFORMER TRAINING - OPTIMIZED FOR RTX 3070Ti")
    print("="*60)
    
    # Load config
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    # Override device
    config['device_id'] = str(device)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Transformer dim: {config['transformer_dim']}")
    print(f"   Attention heads: {config['nhead']}")
    print(f"   Encoder layers: {config['num_encoder_layers']}")
    print(f"   Feed-forward dim: {config['dim_feedforward']}")
    print(f"   Dropout: {config['transformer_dropout']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['lr']}")
    print(f"   Epochs: {config['n_epochs']}")
    
    # Load data
    print("\n📊 Loading UCI-HAR dataset...")
    X_train, y_train = load_inertial_signals(DATASET_PATH, 'train')
    X_test, y_test = load_inertial_signals(DATASET_PATH, 'test')
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    # Create datasets
    train_dataset = UCIHARDataset(X_train, y_train)
    test_dataset = UCIHARDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=config['n_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=config['n_workers'], pin_memory=True)
    
    # Create model
    model = IMUTransformerEncoder(config).to(device)
    print(f"\n📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], 
                                   weight_decay=config['weight_decay'], eps=config['eps'])
    
    # Warmup + Cosine Annealing (critical for transformers)
    warmup_epochs = 10
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, config['n_epochs'])
    
    # Training loop
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    best_f1 = 0.0
    
    print(f"\n🚀 Starting training with {warmup_epochs} warmup epochs...\n")
    
    for epoch in range(1, config['n_epochs'] + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch['imu'] = batch['imu'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping (important for transformer stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch['imu'] = batch['imu'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    
                    outputs = model(batch)
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
            
            warmup_indicator = "🔥 WARMUP" if epoch <= warmup_epochs else ""
            print(f"Epoch {epoch:3d}/{config['n_epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Test Acc: {test_accuracy:.2f}% | F1: {test_f1:.4f} | LR: {current_lr:.6f} {warmup_indicator}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{config['n_epochs']} | "
                      f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | LR: {current_lr:.6f}")
    
    print(f"\n" + "="*60)
    print(f"✅ TRAINING COMPLETE!")
    print(f"="*60)
    print(f"🏆 Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")
    
    # Final evaluation
    print("\n📊 Final Evaluation on Test Set:")
    model.load_state_dict(torch.load(f'{RESULTS_PATH}best_model.pth', weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch['imu'] = batch['imu'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(batch)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                       'SITTING', 'STANDING', 'LAYING']
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=activity_labels))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_labels, yticklabels=activity_labels)
    plt.title(f'CNN-Transformer Confusion Matrix (Accuracy: {best_accuracy:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_PATH}confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model 5: CNN-Transformer Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    test_epochs = [i * 5 for i in range(len(test_accuracies))]
    ax2.plot(test_epochs, test_accuracies, label='Test Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Model 5: CNN-Transformer Training & Test Accuracy')
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
