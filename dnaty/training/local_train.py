"""
dNATY v5 -- ultra-fast local training.
Supports FastDataset (in-RAM tensors) and standard DataLoader.
Optimisations: zero_grad(set_to_none=True), non_blocking, inference_mode, simplified SAM.
"""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dnaty.core.individual import Individual
from dnaty.utils.latency_bench import ONNX_EXPORT_LOCK

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


@contextmanager
def _bn_eval_if_tiny(model: nn.Module, batch_size: int):
    """BatchNorm needs >=2 samples per channel in train mode, otherwise it raises
    "Expected more than 1 value per channel". A leftover batch/chunk of size 1
    (e.g. len(dataset) % batch_size == 1) would crash local_train()/evaluate().

    For those size-<2 forwards we temporarily switch the BatchNorm layers to eval
    mode (running stats) so the pass is well-defined instead of crashing. Affects
    at most one leftover sample per epoch, so the impact on the metric is
    negligible; a no-op when the batch is large enough or the model is in eval."""
    if not (model.training and batch_size < 2):
        yield
        return
    switched = [m for m in model.modules() if isinstance(m, _BN_TYPES) and m.training]
    for m in switched:
        m.eval()
    try:
        yield
    finally:
        for m in switched:
            m.train()


def local_train(
    ind: Individual,
    loader,  # DataLoader or FastDataset
    n_epochs: int = 3,
    lr: float = 1e-3,
    lambda1: float = 1e-4,
    lambda2: float = 1e-3,
    rho: float = 0.05,
    device: str = "cpu",
    batch_size: int = 512,
    augment_images: bool = True,
    mixup_alpha: float = 0.2,
) -> tuple[float, float, float]:
    """
    Train ind for n_epochs. Returns (loss_before, loss_after, mean_grad_norm).
    Supports FastDataset (get_train_batch) and standard DataLoader.
    augment_images: applies RandomCrop+HFlip+mixup to 4D inputs (images). Ignored for MLP (2D).
    mixup_alpha: mixup strength (Beta(a,a)); 0 disables.
    """
    model = ind.model
    if next(model.parameters()).device != torch.device(device):
        model = model.to(device)
        ind.model = model

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7, weight_decay=1e-4)
    # Label smoothing: reduces overfitting, improves generalisation ~0.3-0.5pp
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Structural cost (params/FLOPs) is a constant w.r.t. the weights, so it has
    # zero gradient and cannot be applied as a training-loss term. Compression
    # pressure is applied where it actually bites: the NSGA-II Pareto fitness
    # (see DnatyEvolver._fitness), which trades accuracy against -cost directly.
    # lambda1/lambda2 are kept in the signature for backward compatibility.

    # LR schedule: cosine annealing -- high at start, low at end
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.1)

    is_fast = hasattr(loader, 'get_train_batch')

    # Lazy augmentation: only created when a 4D input (image) is seen. Never fires for MLP (2D).
    augment = None
    def _build_augment(img_size):
        import torchvision.transforms as _T
        return _T.Compose([
            _T.RandomCrop(img_size, padding=4),
            _T.RandomHorizontalFlip(),
        ])

    # Mixed precision (AMP): 2-3x faster on GPU with Tensor Cores, no-op on CPU.
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    use_amp = device_type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True  # auto-tunes conv kernels for fixed shapes
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_first = 0.0
    loss_last  = 0.0
    total_grad_sq = 0.0
    n_steps = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        if is_fast:
            # ceil division: 1000 samples / 512 batch -> 2 batches, not 1
            import math
            n_batches_per_epoch = max(1, math.ceil(loader.n_train / batch_size))
            batches = [loader.get_train_batch(batch_size) for _ in range(n_batches_per_epoch)]
        else:
            batches = loader

        for xb, yb in batches:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            use_mixup = False
            if augment_images and xb.dim() == 4:
                if augment is None:
                    augment = _build_augment(xb.shape[-1])
                xb = augment(xb)
                if mixup_alpha > 0:
                    use_mixup = True
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(xb.size(0), device=xb.device)
                    xb = lam * xb + (1.0 - lam) * xb[perm]
                    yb_b = yb[perm]
            # optimizer.zero_grad() is dynamo-instrumented in torch >= 2.10 and reads a
            # process-global "is exporting" flag; sharing ONNX_EXPORT_LOCK with the export
            # call sites prevents it from racing a concurrent torch.onnx.export() in another
            # thread (see dnaty/utils/latency_bench.py).
            with ONNX_EXPORT_LOCK:
                optimizer.zero_grad(set_to_none=True)
            with _bn_eval_if_tiny(model, xb.size(0)), \
                    torch.autocast(device_type=device_type, enabled=use_amp):
                out = model(xb)
                if use_mixup:
                    loss = lam * criterion(out, yb) + (1.0 - lam) * criterion(out, yb_b)
                else:
                    loss = criterion(out, yb)
            scaler.scale(loss).backward()

            # Unscale before measuring gradient norm (AMP scales grads)
            scaler.unscale_(optimizer)
            with torch.no_grad():
                gn_sq = sum(
                    (p.grad.norm().pow(2) for p in model.parameters() if p.grad is not None),
                    torch.tensor(0.0, device=device),
                )
                total_grad_sq += gn_sq.item()

            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
            n_steps += 1

        avg = epoch_loss / max(n_batches, 1)
        scheduler.step()
        if epoch == 0:
            loss_first = avg
        loss_last = avg

    mean_grad_norm = float(np.sqrt(total_grad_sq / max(n_steps, 1)))
    return loss_first, loss_last, mean_grad_norm


@torch.inference_mode()
def evaluate(
    ind: Individual,
    loader,  # DataLoader or FastDataset
    device: str = "cpu",
    use_train_mode: bool = True,
) -> tuple[float, float]:
    """Return (accuracy, loss). Supports FastDataset and DataLoader.

    use_train_mode=True (default, used during NAS): BatchNorm applies per-batch
    statistics instead of poorly-calibrated running stats (which need 10+
    batches to converge at momentum=0.1). Chunk size 2048 is large enough for
    accurate batch stats. DynamicMLP has no Dropout, so train/eval only differs
    on BatchNorm.

    use_train_mode=False: standard eval() semantics (BN running stats) — this is
    what the deployed/exported model uses, so final reported accuracy should be
    measured this way.
    """
    model = ind.model
    if next(model.parameters()).device != torch.device(device):
        model = model.to(device)
        ind.model = model

    was_training = model.training
    model.train(use_train_mode)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    total_loss = 0.0

    is_fast = hasattr(loader, 'get_val')

    if is_fast:
        vx, vy = loader.get_val()
        # Evaluate in chunks to avoid OOM on VRAM
        chunk = 2048
        for i in range(0, len(vx), chunk):
            xb = vx[i:i+chunk].to(device, non_blocking=True)
            yb = vy[i:i+chunk].to(device, non_blocking=True)
            with _bn_eval_if_tiny(model, xb.size(0)):
                out = model(xb)
            total_loss += criterion(out, yb).item()
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)
    else:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with _bn_eval_if_tiny(model, xb.size(0)):
                out = model(xb)
            total_loss += criterion(out, yb).item()
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)

    model.train(was_training)  # restore caller's mode — evaluation must not leak state
    return correct / max(total, 1), total_loss / max(total, 1)


def micro_adapt(
    ind: Individual,
    loader,
    lr_micro: float = 1e-5,
    top_k_pct: float = 0.03,
    device: str = "cpu",
) -> None:
    """Micro-adaptation: updates the top-k% parameters by gradient magnitude ||dL/dtheta_j||."""
    model = ind.model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    is_fast = hasattr(loader, 'get_train_batch')
    if is_fast:
        x, y = loader.get_train_batch(256)
        x, y = x.to(device), y.to(device)
    else:
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)

    model.zero_grad()
    with _bn_eval_if_tiny(model, x.size(0)):
        out = model(x)
    criterion(out, y).backward()
    with torch.no_grad():
        all_grads = torch.cat([
            p.grad.abs().flatten()
            for p in model.parameters()
            if p.grad is not None
        ])
        if all_grads.numel() == 0:
            return
        k = max(1, int(all_grads.numel() * top_k_pct))
        threshold = all_grads.kthvalue(all_grads.numel() - k).values.item()
        for p in model.parameters():
            if p.grad is not None:
                mask = (p.grad.abs() >= threshold).float()
                p.data -= lr_micro * p.grad * mask
    model.zero_grad()
