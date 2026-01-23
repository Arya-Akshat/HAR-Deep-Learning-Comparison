# Run All Model Tests and Generate Confusion Matrices
# Executes all 4 model tests and creates a comparison summary

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))

def run_all_tests():
    print("="*70)
    print("🚀 HAR DEEP LEARNING COMPARISON - FULL TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test Model 1
    print("\n" + "="*70)
    print("📌 TESTING MODEL 1: CNN-LSTM BASELINE")
    print("="*70)
    from test_model1_cnn_lstm import test_model as test_model1
    acc1, f1_1, cm1 = test_model1()
    results['Model 1: CNN-LSTM'] = {'accuracy': acc1, 'f1': f1_1, 'cm': cm1}
    
    # Test Model 2
    print("\n" + "="*70)
    print("📌 TESTING MODEL 2: CNN-LSTM-ATTENTION")
    print("="*70)
    from test_model2_cnn_lstm_attention import test_model as test_model2
    acc2, f1_2, cm2 = test_model2()
    results['Model 2: CNN-LSTM-Attn'] = {'accuracy': acc2, 'f1': f1_2, 'cm': cm2}
    
    # Test Model 4
    print("\n" + "="*70)
    print("📌 TESTING MODEL 4: CNN-BiLSTM-ATTENTION")
    print("="*70)
    from test_model4_bilstm_attention import test_model as test_model4
    acc4, f1_4, cm4 = test_model4()
    results['Model 4: BiLSTM-Attn'] = {'accuracy': acc4, 'f1': f1_4, 'cm': cm4}
    
    # Test Model 5
    print("\n" + "="*70)
    print("📌 TESTING MODEL 5: CNN-TRANSFORMER")
    print("="*70)
    from test_model5_cnn_transformer import test_model as test_model5
    acc5, f1_5, cm5 = test_model5()
    results['Model 5: Transformer'] = {'accuracy': acc5, 'f1': f1_5, 'cm': cm5}
    
    # Summary
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<30} {'Accuracy':<12} {'F1 Score':<12}")
    print("-"*54)
    for model_name, res in results.items():
        print(f"{model_name:<30} {res['accuracy']:.2f}%{'':<6} {res['f1']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]['accuracy']:.2f}% accuracy")
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 6))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = plt.bar(range(len(models)), accuracies, color=colors)
    plt.xticks(range(len(models)), [m.replace(':', '\n') for m in models], fontsize=9)
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.ylim(min(accuracies) - 5, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # F1 Score comparison
    plt.subplot(1, 2, 2)
    f1_scores = [results[m]['f1'] for m in models]
    bars = plt.bar(range(len(models)), f1_scores, color=colors)
    plt.xticks(range(len(models)), [m.replace(':', '\n') for m in models], fontsize=9)
    plt.ylabel('Macro F1 Score')
    plt.title('F1 Score Comparison')
    plt.ylim(min(f1_scores) - 0.05, 1.0)
    for bar, f1 in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{f1:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    comparison_file = os.path.join(OUTPUT_PATH, 'model_comparison_chart.png')
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison chart saved to: {comparison_file}")
    plt.close()
    
    # Create combined confusion matrix figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    ACTIVITY_LABELS = ['WALKING', 'WALKING_UP', 'WALKING_DOWN', 'SITTING', 'STANDING', 'LAYING']
    cmaps = ['Blues', 'Greens', 'Oranges', 'Purples']
    
    for idx, (model_name, res) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        cm_normalized = res['cm'].astype('float') / res['cm'].sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmaps[idx], 
                    xticklabels=ACTIVITY_LABELS, yticklabels=ACTIVITY_LABELS, ax=ax)
        ax.set_title(f'{model_name}\nAcc: {res["accuracy"]:.2f}% | F1: {res["f1"]:.4f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('HAR Model Comparison - Normalized Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()
    combined_file = os.path.join(OUTPUT_PATH, 'all_confusion_matrices_combined.png')
    plt.savefig(combined_file, dpi=150, bbox_inches='tight')
    print(f"✅ Combined confusion matrices saved to: {combined_file}")
    plt.close()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    
    return results

if __name__ == "__main__":
    run_all_tests()
