import torch
import os
import sys
import json
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('.')
from models.visual_model import get_model
from training.train_visual import get_transforms


def evaluate_model():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = 'data/raw/real_vs_fake/real-vs-fake'
    
    print(f"Device: {DEVICE}")
    
    # Model load
    model = get_model(DEVICE)
    model.load_state_dict(torch.load('models/visual_best.pth', map_location=DEVICE))
    model.eval()
    
    # Test data
    _, val_tf = get_transforms()
    test_ds = datasets.ImageFolder(f'{DATA_DIR}/test', transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"\n📊 Test samples: {len(test_ds)}")
    print(f"📊 Classes: {test_ds.classes}")
    
    y_true = []
    y_pred = []
    y_probs = []
    
    print("\n⏳ Running evaluation...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            probs = torch.softmax(out, dim=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(probs.argmax(dim=1).cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # Metrics
    metrics = {
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, average='macro'), 4),
        'recall':    round(recall_score(y_true, y_pred, average='macro'), 4),
        'f1':        round(f1_score(y_true, y_pred, average='macro'), 4),
        'auc_roc':   round(roc_auc_score(y_true, y_probs), 4),
    }
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k.upper():12}: {v}")
    
    print("\n" + classification_report(y_true, y_pred, target_names=test_ds.classes))
    
    # Confusion matrix
    os.makedirs('outputs', exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_ds.classes, yticklabels=test_ds.classes)
    plt.title('Confusion Matrix — Deepfake Detector')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("💾 Saved: outputs/confusion_matrix.png")
    
    # Save metrics JSON
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("💾 Saved: outputs/evaluation_results.json")
    
    return metrics


if __name__ == "__main__":
    evaluate_model()