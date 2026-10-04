import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import sys
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.visual_model import get_model


# ===== CONFIG (RTX 2050, 4GB VRAM ku tuned) =====
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'data/raw/real_vs_fake/real-vs-fake'
SAVE_DIR = 'models'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    return train_tf, val_tf


def get_loaders():
    train_tf, val_tf = get_transforms()
    
    train_ds = datasets.ImageFolder(f'{DATA_DIR}/train', transform=train_tf)
    val_ds = datasets.ImageFolder(f'{DATA_DIR}/valid', transform=val_tf)
    
    print(f"📊 Train samples: {len(train_ds)}")
    print(f"📊 Val samples:   {len(val_ds)}")
    print(f"📊 Classes:       {train_ds.classes}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return running_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(loader), 100 * correct / total


def main():
    set_seed(42)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("🚀 Starting training...")
    print(f"   Device: {DEVICE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   LR: {LR}")
    
    model = get_model(DEVICE)
    train_loader, val_loader = get_loaders()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    best_acc = 0
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*50}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   LR:         {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{SAVE_DIR}/visual_best.pth')
            print(f"   ✅ Best model saved! ({val_acc:.2f}%)")
        
        torch.save(model.state_dict(), f'{SAVE_DIR}/visual_last.pth')
    
    print(f"\n🎉 Training complete!")
    print(f"   Best Val Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()