"""
dNATY ImageNet Benchmark — Full Scale Validation
Compara dNATY vs ResNet-50, EfficientNet-B0, MobileNetV3

Roda em Colab A100: ~4-6 horas
Local GPU RTX 4090: ~8-10 horas
Local CPU: 🔥 (não recomendado)
"""
from __future__ import annotations
import os, json, sys, time
from pathlib import Path

# Setup Colab
try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount('/content/drive')
    RESULTS_DIR = '/content/drive/My Drive/dNATY_Results'
except ImportError:
    IN_COLAB = False
    RESULTS_DIR = "results_imagenet"

os.makedirs(RESULTS_DIR, exist_ok=True)

if __package__ in (None, ""):
    if not IN_COLAB:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

try:
    from dnaty.experiments.fast_dataset import FastDataset
    from dnaty.core.arch_cnn import DynamicCNN
    from dnaty.core.individual import Individual
    from dnaty.core.memory import EpisodicMemory, Experience
    from dnaty.operators.mutations_cnn import CNN_OPERATORS, apply_cnn_operator
    from dnaty.evolution.selection import nsga2_select
    from dnaty.analysis.stats import summary_stats, paired_ttest
except ImportError:
    print("⚠️  dNATY não encontrado. Execute antes no Colab:")
    print("!git clone https://github.com/pedrovergueiroo/dNATY.git /content/dNATY")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# CONFIG — ImageNet (GPU optimized)
# ═══════════════════════════════════════════════════════════════
SEEDS = [0, 1]
N_GENERATIONS = 15  # ImageNet: mais gerações
N_POP = 12          # Menor que CIFAR (memory constraints)
T_LOCAL = 8         # Mais épocas (ImageNet precisa)
BATCH_SIZE = 256    # ImageNet padrão
IMAGENET_SUBSET = 100000  # 100K treino (10% de 1.2M) — viável em Colab
N_CLASSES = 1000

# Data augmentation para ImageNet
TRAIN_TRANSFORMS = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomRotation(20),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORMS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ── Baselines para ImageNet ────────────────────────────────────
class ResNet50(nn.Module):
    """ResNet-50 padrão para ImageNet"""
    def __init__(self, n_classes=1000):
        super().__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        return self.model(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 para ImageNet"""
    def __init__(self, n_classes=1000):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = nn.Linear(1280, n_classes)

    def forward(self, x):
        return self.model(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class MobileNetV3(nn.Module):
    """MobileNetV3-Large para ImageNet"""
    def __init__(self, n_classes=1000):
        super().__init__()
        self.model = models.mobilenet_v3_large(pretrained=False)
        self.model.classifier[1] = nn.Linear(1280, n_classes)

    def forward(self, x):
        return self.model(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def load_imagenet_data(subset_size=100000, device='cuda'):
    """Carrega ImageNet localmente ou do Colab"""
    if IN_COLAB:
        # Colab: ImageNet já disponível
        imagenet_path = '/content/drive/My Drive/ImageNet'
    else:
        imagenet_path = '/data/ImageNet'

    if not os.path.exists(imagenet_path):
        print(f"⚠️  ImageNet não encontrado em {imagenet_path}")
        print("No Colab: Download ImageNet para sua Google Drive")
        return None, None

    print(f"Carregando ImageNet subset ({subset_size} imagens)...")

    train_dataset = ImageNet(imagenet_path, split='train', transform=TRAIN_TRANSFORMS)
    val_dataset = ImageNet(imagenet_path, split='val', transform=VAL_TRANSFORMS)

    # Subset para viabilidade
    if subset_size < len(train_dataset):
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)

    return train_loader, val_loader


def evaluate_model(model, val_loader, device, top_k=(1, 5)):
    """Avalia Top-1 e Top-5 accuracy no ImageNet"""
    model = model.to(device)
    model.eval()

    top1_correct = top5_correct = total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            # Top-1
            _, preds1 = outputs.topk(1, dim=1)
            top1_correct += (preds1.squeeze() == labels).sum().item()

            # Top-5
            _, preds5 = outputs.topk(5, dim=1)
            top5_correct += sum(labels[i] in preds5[i] for i in range(len(labels)))

            total += len(labels)

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    return top1_acc, top5_acc


def train_model(model, train_loader, val_loader, n_epochs, lr, device, model_name="Model"):
    """Treina modelo genérico com SAM e Data Augmentation"""
    model = model.to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.01)

    rho = 0.05  # SAM radius

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # SAM Step 1
            opt.zero_grad(set_to_none=True)
            loss = crit(model(images), labels)
            loss.backward()

            # SAM Step 2 - Perturbação
            with torch.no_grad():
                grad_norm = sum(p.grad.norm().pow(2) for p in model.parameters() if p.grad is not None).sqrt()
                scale = rho / (grad_norm + 1e-12)
                for p in model.parameters():
                    if p.grad is not None:
                        p.data_orig = p.data.clone()
                        p.data.add_(p.grad, alpha=scale)

            # SAM Step 3 - Recomputar loss
            opt.zero_grad(set_to_none=True)
            loss_perturbed = crit(model(images), labels)
            loss_perturbed.backward()

            # SAM Step 4 - Restaurar e update
            with torch.no_grad():
                for p in model.parameters():
                    if hasattr(p, 'data_orig'):
                        p.data.copy_(p.data_orig)

            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        top1, top5 = evaluate_model(model, val_loader, device)
        print(f"  [{model_name}] Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Top1: {top1*100:.2f}% | Top5: {top5*100:.2f}%")

    return evaluate_model(model, val_loader, device)


def run_baseline_experiments(train_loader, val_loader, device):
    """Avalia baselines (ResNet-50, EfficientNet-B0, MobileNetV3)"""
    results = {}

    baselines = {
        'ResNet-50': ResNet50(N_CLASSES),
        'EfficientNet-B0': EfficientNetB0(N_CLASSES),
        'MobileNetV3': MobileNetV3(N_CLASSES),
    }

    for name, model in baselines.items():
        print(f"\nTreinando {name}...")
        top1, top5 = train_model(
            model, train_loader, val_loader,
            n_epochs=12, lr=0.1, device=device, model_name=name
        )
        results[name] = {
            'top1': round(top1, 4),
            'top5': round(top5, 4),
            'params': model.count_params()
        }
        print(f"  {name}: Top1={top1*100:.2f}%, Top5={top5*100:.2f}%, Params={model.count_params():,}")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n{'='*70}")
    print("dNATY ImageNet Benchmark — Full Validation")
    print(f"{'='*70}")

    # Load data
    train_loader, val_loader = load_imagenet_data(IMAGENET_SUBSET, device)

    if train_loader is None:
        print("Abortando. ImageNet não disponível.")
        return

    # Run baselines
    print(f"\n{'─'*70}")
    print("BASELINE MODELS (ResNet-50, EfficientNet-B0, MobileNetV3)")
    print(f"{'─'*70}")
    baseline_results = run_baseline_experiments(train_loader, val_loader, device)

    # Results
    all_results = {
        "ImageNet": {
            "subset_size": IMAGENET_SUBSET,
            "baselines": baseline_results,
            "dnaty_status": "Em desenvolvimento",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    out_path = os.path.join(RESULTS_DIR, "imagenet_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Resultados salvos em: {out_path}")

    # Print summary
    print(f"\n{'─'*70}")
    print("RESUMO BASELINE ImageNet")
    print(f"{'─'*70}")
    for name, metrics in baseline_results.items():
        print(f"{name:18} | Top1: {metrics['top1']*100:6.2f}% | Top5: {metrics['top5']*100:6.2f}% | Params: {metrics['params']:,}")


if __name__ == "__main__":
    main()
