#!/usr/bin/env python3
"""
Teste real do compress() -- 60K amostras MNIST, verifica TUDO.

Checks:
  1. compress() roda sem erros
  2. FLOPs reduction >= 20% (provado: ~46%)
  3. Accuracy >= 95% no modelo comprimido
  4. Modelo comprimido tem menos params que o original
  5. Inferencia funciona (forward pass real)
  6. Export ONNX funciona
  7. Reload do .pt funciona
  8. claude_service retorna texto (template ou API)
"""
from __future__ import annotations
import sys, time, os, tempfile
from pathlib import Path

# Fix Windows charmap issue: torch ONNX exporter logs emojis
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from dnaty import compress, CompressResult
from dnaty.core.arch import DynamicMLP
from dnaty.experiments.fast_dataset import FastDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASS = "[OK]"
FAIL = "[FAIL]"

results: list[tuple[str, bool, str]] = []

def check(name: str, condition: bool, detail: str = ""):
    mark = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    return condition


print(f"\n{'='*65}")
print(f"  dNATY compress() — TESTE REAL (MNIST 60K, {DEVICE.upper()})")
print(f"{'='*65}\n")

# ── 1. Carrega dataset completo (60K) ─────────────────────────────────────────
print("[1] Carregando MNIST 60K ...")
t0 = time.perf_counter()
ds = FastDataset("MNIST", device=DEVICE, train_subset=60_000)
print(f"    train={ds.n_train:,}  val={len(ds.get_val()[0]):,}  ({time.perf_counter()-t0:.1f}s)\n")

check("Dataset carregado", ds.n_train >= 50_000, f"n_train={ds.n_train:,}")

# ── 2. Modelo inicial grande ──────────────────────────────────────────────────
print("[2] Modelo original (MLP [784,512,256,128]->10) ...")
orig_model = DynamicMLP(
    layer_sizes=[784, 512, 256, 128],
    activations=["relu", "relu", "relu"],
    n_classes=10,
).to(DEVICE)
orig_params = sum(p.numel() for p in orig_model.parameters())
print(f"    params={orig_params:,}\n")

# ── 3. Roda compress() ────────────────────────────────────────────────────────
print("[3] Rodando compress() — 30 geracoes, pop=15 ...")
t0 = time.perf_counter()
result: CompressResult = compress(
    orig_model,
    ds,
    target_flops=0.5,
    n_generations=30,
    n_pop=15,
    device=DEVICE,
    verbose=True,
    seed=42,
)
elapsed = time.perf_counter() - t0
print(f"\n    Tempo: {elapsed:.0f}s\n")

# ── 4. Verifica metricas ──────────────────────────────────────────────────────
print("[4] Verificando metricas ...")
check("Sem erros (compress rodou)", True)
check("FLOPs reduction >= 20%",
      result.flops_reduction >= 0.20,
      f"{result.flops_reduction_pct:.1f}%")
check("Accuracy >= 95%",
      result.accuracy >= 0.95,
      f"{result.accuracy*100:.2f}%")
check("Params comprimido < original",
      result.compressed_params < orig_params,
      f"{orig_params:,} -> {result.compressed_params:,}")
check("CompressResult.summary() nao vazio",
      bool(result.summary()),
      result.summary())

# ── 5. Forward pass real ──────────────────────────────────────────────────────
print("\n[5] Forward pass no modelo comprimido ...")
result.model.eval()
with torch.inference_mode():
    x_test = torch.randn(32, 784, device=DEVICE)
    try:
        logits = result.model(x_test)
        preds  = logits.argmax(dim=1)
        check("Forward pass (batch=32)", logits.shape == (32, 10),
              f"output shape={tuple(logits.shape)}")
        check("Predicoes validas (0-9)",
              preds.min().item() >= 0 and preds.max().item() <= 9,
              f"range=[{preds.min().item()},{preds.max().item()}]")
    except Exception as e:
        check("Forward pass", False, str(e))

# ── 6. Avalia no val set real ─────────────────────────────────────────────────
print("\n[6] Avaliando no validation set completo ...")
from dnaty.training.local_train import evaluate
from dnaty.core.individual import Individual
val_ind = Individual(result.model)
acc_val, loss_val = evaluate(val_ind, ds, device=DEVICE)
check("Val accuracy >= 95% (no val set)",
      acc_val >= 0.95,
      f"acc={acc_val*100:.2f}%  loss={loss_val:.4f}")

# ── 7. Salva e recarrega .pt ──────────────────────────────────────────────────
print("\n[7] Salva e recarrega pesos (.pt) ...")
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    pt_path = f.name
try:
    torch.save(result.model.state_dict(), pt_path)

    # Reconstroi a partir do arch
    arch   = result.arch
    # arch is hidden-only (input 784 excluded); DynamicMLP takes [input] + hidden
    reload = DynamicMLP(
        layer_sizes=[784] + arch,
        activations=["relu"] * len(arch),
        n_classes=10,
    )
    reload.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    reload.eval()
    with torch.inference_mode():
        out_r = reload(torch.randn(1, 784))
    check("Salva .pt e recarrega", out_r.shape == (1, 10), f"arch={arch}")
except Exception as e:
    check("Salva .pt e recarrega", False, str(e))
finally:
    os.unlink(pt_path)

# ── 8. Export ONNX ────────────────────────────────────────────────────────────
print("\n[8] Export ONNX ...")
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    onnx_path = f.name
try:
    dummy = torch.randn(1, 784)
    result.model.cpu().eval()
    torch.onnx.export(
        result.model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=18,
    )
    size_kb = Path(onnx_path).stat().st_size / 1024
    check("Export ONNX", Path(onnx_path).stat().st_size > 0, f"{size_kb:.1f} KB")
except Exception as e:
    check("Export ONNX", False, str(e))
finally:
    os.unlink(onnx_path)

# ── 9. Claude service (template fallback) ────────────────────────────────────
print("\n[9] Claude service (template fallback) ...")
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dnaty_saas"))
    from services.claude_service import explain_compression
    exp, code = explain_compression({
        "original_params":   orig_params,
        "compressed_params": result.compressed_params,
        "accuracy":          result.accuracy,
        "flops_reduction":   result.flops_reduction,
        "arch":              result.arch,
        "input_size":        784,
        "n_classes":         10,
    })
    check("Explanation gerada",  len(exp) > 50,  exp[:80] + "...")
    check("Codigo de deploy gerado", "DynamicMLP" in code and "torch" in code)
except Exception as e:
    check("Claude service", False, str(e))

# ── Resultado final ───────────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
all_ok = passed == total

print(f"\n{'='*65}")
print(result.summary())
print(f"{'='*65}")
print(f"\n  {passed}/{total} checks passaram  |  tempo total: {elapsed:.0f}s")

if all_ok:
    print(f"\n  TUDO OK — dNATY compress() funciona de ponta a ponta.\n")
else:
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n  FALHOU: {', '.join(failed)}\n")

sys.exit(0 if all_ok else 1)
