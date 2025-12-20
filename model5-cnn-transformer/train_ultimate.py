# CNN-Transformer ULTIMATE Training - Maximum Performance
# Larger model, longer training, advanced techniques

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
import math

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.IMUTransformerEncoder import IMUTransformerEncoder

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/human+activity+recognition+using+smartphones/UCI HAR Dataset/"
RESULTS_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/results/model5-cnn-transformer/"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_ultimate.json")

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
# Data Loading with Augmentation
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
    """UCI-HAR Dataset with optional augmentation"""
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        
        if self.augment and self.training_mode:
            # Random noise augmentation
            if np.random.random() < 0.5:
                noise = torch.randn_like(x) * 0.02
                x = x + noise
            
            # Random time shift
            if np.random.random() < 0.3:
                shift = np.random.randint(-5, 5)
                x = torch.roll(x, shifts=shift, dims=0)
        
        return {'imu': x, 'label': self.y[idx]}
    
    def set_training_mode(self, mode):
        self.training_mode = mode

# ============================================================================
# Cosine Annealing with Warm Restarts
# ============================================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
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
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = max(self.base_lrs[i] * lr_scale, self.min_lr)
        
        return self.optimizer.param_groups[0]['lr']

# ============================================================================
# Label Smoothing Cross Entropy
# ============================================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)
        
        # Create smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))

# ============================================================================
# Training Function
# ============================================================================
def train_model():
    print("\n" + "="*70)
    print("🚀 CNN-TRANSFORMER ULTIMATE - MAXIMUM PERFORMANCE")
    print("="*70)
    
    # Load config
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    # Override device
    config['device_id'] = str(device)
    
    print(f"\n⚙️  Configuration (ULTIMATE):")
    print(f"   Transformer dim: {config['transformer_dim']} (larger)")
    print(f"   Attention heads: {config['nhead']}")
    print(f"   Encoder layers: {config['num_encoder_layers']} (deeper)")
    print(f"   Feed-forward dim: {config['dim_feedforward']}")
    print(f"   Dropout: {config['transformer_dropout']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['lr']}")
    print(f"   Weight decay: {config['weight_decay']}")
    print(f"   Label smoothing: {config['label_smoothing']}")
    print(f"   Warmup epochs: {config['warmup_epochs']}")
    print(f"   Total epochs: {config['n_epochs']}")
    
    # Load data
    print("\n📊 Loading UCI-HAR dataset...")
    X_train, y_train = load_inertial_signals(DATASET_PATH, 'train')
    X_test, y_test = load_inertial_signals(DATASET_PATH, 'test')
    
    print(f"   Training samples: {X_train.shape}")
    print(f"   Test samples: {X_test.shape}")
    
    # Create datasets with augmentation for training
    train_dataset = UCIHARDataset(X_train, y_train, augment=True)
    train_dataset.set_training_mode(True)
    test_dataset = UCIHARDataset(X_test, y_test, augment=False)
    test_dataset.training_mode = False
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    model = IMUTransformerEncoder(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Total parameters: {total_params:,}")
    
    # Loss with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=config['warmup_epochs'], 
        total_epochs=config['n_epochs']
    )
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_f1_scores = []
    best_accuracy = 0.0
    best_f1 = 0.0
    best_epoch = 0
    patience = 50  # Early stopping patience
    no_improve_count = 0
    
    warmup_epochs = config['warmup_epochs']
    gradient_clip = config['gradient_clip']
    
    print(f"\n🚀 Starting training (this will take ~30-40 minutes)...")
    print(f"   Early stopping patience: {patience} epochs\n")
    
    for epoch in range(1, config['n_epochs'] + 1):
        # Training
        model.train()
        train_dataset.set_training_mode(True)
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Move batch to device
            batch_device = {'imu': batch['imu'].to(device)}
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_device)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluation every 5 epochs
        if epoch % 5 == 0 or epoch <= 20:
            model.eval()
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch_device = {'imu': batch['imu'].to(device)}
                    labels = batch['label'].to(device)
                    outputs = model(batch_device)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            test_accuracy = 100 * correct / total
            test_f1 = f1_score(all_labels, all_preds, average='weighted')
            test_accuracies.append(test_accuracy)
            test_f1_scores.append(test_f1)
            
            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_f1 = test_f1
                best_epoch = epoch
                best_preds = all_preds.copy()
                best_labels = all_labels.copy()
                torch.save(model.state_dict(), f'{RESULTS_PATH}best_model.pth')
                no_improve_count = 0
            else:
                no_improve_count += 5
            
            warmup_indicator = "🔥 WARMUP" if epoch <= warmup_epochs else ""
            improve_indicator = "⭐ NEW BEST!" if test_accuracy >= best_accuracy and no_improve_count == 0 else ""
            
            print(f"Epoch {epoch:3d}/{config['n_epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Test Acc: {test_accuracy:.2f}% | F1: {test_f1:.4f} | LR: {current_lr:.6f} {warmup_indicator} {improve_indicator}")
            
            # Early stopping check
            if no_improve_count >= patience and epoch > warmup_epochs + 50:
                print(f"\n⚠️ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        else:
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}/{config['n_epochs']} | "
                      f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | LR: {current_lr:.6f}")
    
    # Final evaluation with best model
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"🏆 Best Test Accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(f'{RESULTS_PATH}best_model.pth', weights_only=True))
    model.eval()
    
    # Classification report
    activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    print(f"\n📋 Classification Report:")
    print(classification_report(best_labels, best_preds, target_names=activity_labels))
    
    # Confusion matrix
    cm = confusion_matrix(best_labels, best_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_labels, yticklabels=activity_labels)
    plt.title(f'Model 5: CNN-Transformer Ultimate\nAccuracy: {best_accuracy:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_PATH}confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model 5: CNN-Transformer Ultimate - Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accuracies, label='Train Accuracy', alpha=0.7)
    test_epochs = [i * 5 for i in range(1, len(test_accuracies) + 1)]
    # Adjust for epochs <= 20 logged individually
    test_epochs = []
    ep = 0
    for i in range(len(test_accuracies)):
        if ep < 20:
            ep += 1
            if ep <= 20:
                test_epochs.append(ep)
        else:
            ep += 5
            test_epochs.append(ep)
    # Simpler approach - just use indices
    test_epochs = list(range(1, len(test_accuracies) + 1))
    ax2.plot(test_epochs, test_accuracies, label='Test Accuracy', marker='o', markersize=3)
    ax2.axhline(y=best_accuracy, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_accuracy:.2f}%')
    ax2.set_xlabel('Evaluation Point')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Model 5: CNN-Transformer Ultimate - Accuracy')
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
