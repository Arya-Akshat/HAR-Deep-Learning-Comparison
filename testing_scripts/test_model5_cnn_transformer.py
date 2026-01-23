# Testing Script for Model 5: CNN-Transformer (Ultimate)
# Loads trained model and generates confusion matrix

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# ============================================================================
# CONFIGURATION (EXACT SAME AS train_ultimate.py)
# ============================================================================
DATASET_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/human+activity+recognition+using+smartphones/UCI HAR Dataset/"
MODEL_PATH = "D:/Coding/Group/HAR-Deep-Learning-Comparison/results/model5-cnn-transformer/best_model.pth"
OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))

# Config from config_ultimate.json
CONFIG = {
    "name": "UCI-HAR Ultimate Transformer",
    "input_dim": 9,
    "num_classes": 6,
    "window_size": 128,
    "transformer_dim": 256,
    "nhead": 8,
    "num_encoder_layers": 4,
    "dim_feedforward": 512,
    "transformer_dropout": 0.3,
    "transformer_activation": "gelu",
    "encode_position": True,
    "cls_head": "cls_token",
    "batch_size": 64,
}

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
# Model Architecture (EXACT from models/IMUTransformerEncoder.py)
# ============================================================================
class IMUTransformerEncoder(nn.Module):

    def __init__(self, config):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = config.get("transformer_dim")

        self.input_proj = nn.Sequential(nn.Conv1d(config.get("input_dim"), self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())

        self.window_size = config.get("window_size")
        self.encode_position = config.get("encode_position")
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = config.get("nhead"),
                                       dim_feedforward = config.get("dim_feedforward"),
                                       dropout = config.get("transformer_dropout"),
                                       activation = config.get("transformer_activation"))

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = config.get("num_encoder_layers"),
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        num_classes = config.get("num_classes")
        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, self.transformer_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim//4, num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        src = data.get('imu')  # Shape N x S x C with S = sequence length, N = batch size, C = channels

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed

        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]

        # Class probability
        target = self.log_softmax(self.imu_head(target))
        return target

# ============================================================================
# Data Loading (EXACT from train_ultimate.py)
# Note: Model 5 uses (samples, 128, 9) - sequence first
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
    """UCI-HAR Dataset for testing (no augmentation)"""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'imu': self.X[idx], 'label': self.y[idx]}

# ============================================================================
# Testing and Confusion Matrix Generation
# ============================================================================
def test_model():
    print("\n" + "="*70)
    print("MODEL 5: CNN-TRANSFORMER (ULTIMATE) - TESTING")
    print("="*70)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Transformer dim: {CONFIG['transformer_dim']}")
    print(f"   Attention heads: {CONFIG['nhead']}")
    print(f"   Encoder layers: {CONFIG['num_encoder_layers']}")
    print(f"   Feed-forward dim: {CONFIG['dim_feedforward']}")
    print(f"   Dropout: {CONFIG['transformer_dropout']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    
    # Load test data
    print("\n📊 Loading UCI-HAR test dataset...")
    X_test, y_test = load_inertial_signals(DATASET_PATH, 'test')
    print(f"   Test samples: {X_test.shape}")
    
    # Create dataset and loader
    test_dataset = UCIHARDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model and load weights
    model = IMUTransformerEncoder(CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Total parameters: {total_params:,}")
    
    print(f"\n📂 Loading model from: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # Inference
    all_preds = []
    all_labels = []
    
    print("\n🔄 Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            batch_device = {'imu': batch['imu'].to(device)}
            labels = batch['label'].to(device)
            outputs = model(batch_device)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\n" + "="*70)
    print(f"📊 TEST RESULTS")
    print(f"="*70)
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Macro F1 Score: {f1_macro:.4f}")
    print(f"   Weighted F1 Score: {f1_weighted:.4f}")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=ACTIVITY_LABELS))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=ACTIVITY_LABELS, yticklabels=ACTIVITY_LABELS)
    plt.title(f'Model 5: CNN-Transformer (Ultimate)\nAccuracy: {accuracy:.2f}% | F1: {f1_macro:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_PATH, 'confusion_matrix_model5_cnn_transformer.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to: {output_file}")
    plt.close()
    
    # Also save normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Purples', 
                xticklabels=ACTIVITY_LABELS, yticklabels=ACTIVITY_LABELS)
    plt.title(f'Model 5: CNN-Transformer (Ultimate) - Normalized\nAccuracy: {accuracy:.2f}% | F1: {f1_macro:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_file_norm = os.path.join(OUTPUT_PATH, 'confusion_matrix_model5_cnn_transformer_normalized.png')
    plt.savefig(output_file_norm, dpi=150, bbox_inches='tight')
    print(f"✅ Normalized confusion matrix saved to: {output_file_norm}")
    plt.close()
    
    print(f"\n" + "="*70)
    print(f"✅ TESTING COMPLETE!")
    print(f"="*70)
    
    return accuracy, f1_macro, cm

if __name__ == "__main__":
    test_model()
