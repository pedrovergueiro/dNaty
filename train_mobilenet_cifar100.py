#!/usr/bin/env python3
"""Quick MobileNetV3 CIFAR-100 benchmark - correção do erro"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Dataset
print("Carregando CIFAR-100...")
train_transforms = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

val_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = CIFAR100('/tmp/cifar100', train=True, download=True, transform=train_transforms)
val_dataset = CIFAR100('/tmp/cifar100', train=False, download=True, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

print(f"[OK] Train: {len(train_loader)} batches | Val: {len(val_loader)} batches\n")

# MobileNetV3Large CORRIGIDO
class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = models.mobilenet_v3_large(weights=None)
        # CORREÇÃO: Reconstruir classifier (original: 1000 → 100)
        self.model.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        return self.model(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# Evaluate
def evaluate(model, val_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total

# Train
print("="*70)
print("TREINANDO MobileNetV3-Large em CIFAR-100")
print("="*70)

model = MobileNetV3Large(100)
model = model.to(device)
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25, eta_min=0.001)

best_acc = 0
t0 = time.time()

for epoch in range(25):
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/25", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        loss = crit(model(images), labels)
        loss.backward()
        opt.step()

    scheduler.step()

    acc = evaluate(model, val_loader, device)
    best_acc = max(best_acc, acc)
    print(f"  Epoch {epoch+1}/25 | Acc: {acc*100:6.2f}% | Best: {best_acc*100:6.2f}%")

elapsed = time.time() - t0
final_acc = evaluate(model, val_loader, device)

print("\n" + "="*70)
print("RESULTADO FINAL")
print("="*70)
print(f"Acurácia: {final_acc*100:.2f}%")
print(f"Parâmetros: {model.count_params():,}")
print(f"Tempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print("="*70)

# Salva resultado
result = {
    'model': 'MobileNetV3-Large',
    'dataset': 'CIFAR-100',
    'accuracy': round(final_acc, 4),
    'best_accuracy': round(best_acc, 4),
    'params': model.count_params(),
    'time_s': round(elapsed, 1),
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
}

with open('mobilenet_cifar100_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n[OK] Resultado salvo em: mobilenet_cifar100_result.json")
