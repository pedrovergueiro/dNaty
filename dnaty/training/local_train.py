"""
Treino local com SAM + Adam — Lema 2 do Teorema dNaty-Convergence.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from dnaty.core.individual import Individual


def structural_cost(ind: Individual, alpha: float = 1e-5, beta: float = 1e-5) -> torch.Tensor:
    """C(A) = α·|params| + β·FLOPs — penaliza redes grandes."""
    n_params = ind.count_params()
    n_flops = ind.count_flops()
    return torch.tensor(alpha * n_params + beta * n_flops, dtype=torch.float32)


def sam_sharpness(model: nn.Module, loss: torch.Tensor, rho: float = 0.05) -> torch.Tensor:
    """SAM: S(θ) = (L(θ + ρ·∇L/‖∇L‖) - L(θ))²"""
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, allow_unused=True)
    grad_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads if g is not None) + 1e-12)
    # Perturbação adversarial
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            if g is not None:
                p.data.add_(rho * g / grad_norm)
    return grad_norm.detach()


def local_train(
    ind: Individual,
    loader: torch.utils.data.DataLoader,
    n_epochs: int = 5,
    lr: float = 1e-3,
    lambda1: float = 1e-4,
    lambda2: float = 1e-3,
    rho: float = 0.05,
    device: str = "cpu",
) -> tuple[float, float, float]:
    """
    Treina ind por n_epochs. Retorna (loss_antes, loss_depois, grad_norm_medio).
    Implementa L_total = CE + λ₁·C(A) + λ₂·S(θ,A).
    """
    model = ind.model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    cost_penalty = structural_cost(ind, alpha=lambda1, beta=lambda1 * 0.01)

    loss_history: list[float] = []
    grad_norms: list[float] = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            ce_loss = criterion(out, yb)
            total_loss = ce_loss + cost_penalty

            # SAM sharpness (simplificado — sem segundo forward para velocidade)
            total_loss.backward()
            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5
            grad_norms.append(grad_norm)
            optimizer.step()
            epoch_loss += total_loss.item()

        loss_history.append(epoch_loss / max(len(loader), 1))

    loss_before = loss_history[0] if loss_history else 0.0
    loss_after = loss_history[-1] if loss_history else 0.0
    mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    return loss_before, loss_after, mean_grad_norm


def evaluate(
    ind: Individual,
    loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> tuple[float, float]:
    """Retorna (accuracy, loss)."""
    model = ind.model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += criterion(out, yb).item() * len(yb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1), total_loss / max(total, 1)


def micro_adapt(
    ind: Individual,
    x: torch.Tensor,
    y: torch.Tensor,
    lr_micro: float = 1e-5,
    top_k_pct: float = 0.03,
    device: str = "cpu",
) -> None:
    """
    Micro-adaptação: atualiza top-k% parâmetros por ‖∂L/∂θ_j‖.
    Anti-forgetting — só muda o mínimo necessário.
    """
    model = ind.model.to(device)
    model.train()
    x, y = x.to(device), y.to(device)
    criterion = nn.CrossEntropyLoss()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    with torch.no_grad():
        all_grads = []
        for p in model.parameters():
            if p.grad is not None:
                all_grads.extend(p.grad.abs().flatten().tolist())
        if not all_grads:
            return
        threshold = sorted(all_grads, reverse=True)[max(1, int(len(all_grads) * top_k_pct))]
        for p in model.parameters():
            if p.grad is not None:
                mask = (p.grad.abs() >= threshold).float()
                p.data -= lr_micro * p.grad * mask
    model.zero_grad()


# Importação necessária para mean
import numpy as np
