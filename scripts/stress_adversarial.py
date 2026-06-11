"""
Adversarial stress suite for dNATY — hunts crashes, bad error messages,
and contract inconsistencies. Tests the repo source, tiny budgets everywhere.
Output: one line per scenario  [PASS|FAIL|CRASH]  + summary.
"""
import sys, io, os, json, time, math, tempfile, traceback, warnings, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONUTF8", "1")
TMP = tempfile.gettempdir()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import dnaty
from dnaty import compress, compress_cnn, compress_with_backbone, prune_conv_channels, load as dnaty_load
from dnaty.utils.flops_counter import count_flops, flops_by_layer
from dnaty.monitoring import DriftDetector, ProductionTracker

RESULTS = []

def scenario(name):
    def deco(fn):
        def run():
            t0 = time.time()
            buf = io.StringIO()
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    with contextlib.redirect_stdout(buf):
                        fn(w)
                RESULTS.append(("PASS", name, f"{time.time()-t0:.1f}s", ""))
            except AssertionError as e:
                RESULTS.append(("FAIL", name, f"{time.time()-t0:.1f}s", str(e)[:200]))
            except Exception as e:
                tb = traceback.format_exc().strip().splitlines()
                RESULTS.append(("CRASH", name, f"{time.time()-t0:.1f}s", f"{type(e).__name__}: {str(e)[:160]} @ {tb[-2].strip()[:90]}"))
        run.__name__ = name
        return run
    return deco

def mk_data(n=300, feats=16, classes=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, feats, generator=g)
    y = (X[:, 0] > 0).long() if classes == 2 else torch.randint(0, classes, (n,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

def mk_model(feats=16, hidden=(64, 32), classes=2):
    layers, prev = [], feats
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, classes))
    return nn.Sequential(*layers)

FAST = dict(n_generations=2, n_pop=3, verbose=False, finetune_epochs=0)

# ── A. Degenerate data ────────────────────────────────────────────────────────

@scenario("A1 two samples total")
def t_a1(w):
    X = torch.randn(2, 8); y = torch.tensor([0, 1])
    r = compress(mk_model(8), DataLoader(TensorDataset(X, y), batch_size=2), **FAST)
    assert r.model is not None

@scenario("A2 single class in labels")
def t_a2(w):
    X = torch.randn(100, 8); y = torch.zeros(100, dtype=torch.long)
    r = compress(mk_model(8), DataLoader(TensorDataset(X, y), batch_size=32), **FAST)
    assert r.model is not None

@scenario("A3 100 classes / 500 samples")
def t_a3(w):
    X = torch.randn(500, 16); y = torch.randint(0, 100, (500,))
    r = compress(mk_model(16, classes=100), DataLoader(TensorDataset(X, y), batch_size=64), **FAST)
    assert r.model is not None

@scenario("A4 extreme imbalance 99:1")
def t_a4(w):
    X = torch.randn(1000, 8); y = torch.cat([torch.zeros(990), torch.ones(10)]).long()
    r = compress(mk_model(8), DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True), **FAST)
    assert r.model is not None

@scenario("A5 NaN in features -> useful error or survives")
def t_a5(w):
    X = torch.randn(200, 8); X[5, 3] = float("nan"); y = (X[:, 0] > 0).long()
    r = compress(mk_model(8), DataLoader(TensorDataset(X, y), batch_size=64), **FAST)
    acc = r.accuracy
    assert acc == acc, f"accuracy is NaN ({acc}) — NaN propagated silently, no warning/error"

@scenario("A6 Inf in features -> useful error or survives")
def t_a6(w):
    X = torch.randn(200, 8); X[7, 2] = float("inf"); y = (X[:, 0] > 0).long()
    r = compress(mk_model(8), DataLoader(TensorDataset(X, y), batch_size=64), **FAST)
    assert r.accuracy == r.accuracy and r.accuracy != float("inf")

@scenario("A7 all-constant features")
def t_a7(w):
    X = torch.zeros(200, 8); y = torch.randint(0, 2, (200,))
    r = compress(mk_model(8), DataLoader(TensorDataset(X, y), batch_size=64), **FAST)
    assert r.model is not None

@scenario("A8 labels {5,6} but model outputs 2 -> clear error expected")
def t_a8(w):
    X = torch.randn(100, 8); y = torch.randint(5, 7, (100,))
    try:
        compress(mk_model(8, classes=2), DataLoader(TensorDataset(X, y), batch_size=32), **FAST)
        raise AssertionError("accepted out-of-range labels silently (no error)")
    except AssertionError:
        raise
    except Exception as e:
        msg = str(e).lower()
        assert "class" in msg or "target" in msg or "label" in msg or "index" in msg, \
            f"error is cryptic: {type(e).__name__}: {str(e)[:120]}"

@scenario("A9 float64 features")
def t_a9(w):
    X = torch.randn(200, 8, dtype=torch.float64); y = (X[:, 0] > 0).long()
    r = compress(mk_model(8), DataLoader(TensorDataset(X, y), batch_size=64), **FAST)
    assert r.model is not None

@scenario("A10 float labels (should error clearly or cast)")
def t_a10(w):
    X = torch.randn(200, 8); y = (X[:, 0] > 0).float()
    try:
        r = compress(mk_model(8), DataLoader(TensorDataset(X, y), batch_size=64), **FAST)
        assert r.model is not None
    except Exception as e:
        msg = str(e).lower()
        assert "long" in msg or "int" in msg or "type" in msg or "dtype" in msg, \
            f"error is cryptic: {type(e).__name__}: {str(e)[:120]}"

# ── B. Degenerate models ──────────────────────────────────────────────────────

@scenario("B1 single Linear, no hidden")
def t_b1(w):
    r = compress(nn.Sequential(nn.Linear(16, 2)), mk_data(), **FAST)
    assert r.model is not None

@scenario("B2 model with Dropout+BatchNorm")
def t_b2(w):
    m = nn.Sequential(nn.Linear(16, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))
    r = compress(m, mk_data(), **FAST)
    assert r.model is not None

@scenario("B3 Linear bias=False")
def t_b3(w):
    m = nn.Sequential(nn.Linear(16, 64, bias=False), nn.ReLU(), nn.Linear(64, 2, bias=False))
    r = compress(m, mk_data(), **FAST)
    assert r.model is not None

@scenario("B4 empty Sequential -> clean error expected")
def t_b4(w):
    try:
        compress(nn.Sequential(), mk_data(), **FAST)
        raise AssertionError("accepted an empty model silently")
    except AssertionError:
        raise
    except Exception as e:
        msg = str(e).lower()
        assert "linear" in msg or "layer" in msg or "model" in msg or "empty" in msg, \
            f"error is cryptic: {type(e).__name__}: {str(e)[:120]}"

@scenario("B5 conv model into compress() -> clean error expected")
def t_b5(w):
    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8*30*30, 2))
    X = torch.randn(40, 3, 32, 32); y = torch.randint(0, 2, (40,))
    try:
        compress(m, DataLoader(TensorDataset(X, y), batch_size=16), **FAST)
    except Exception as e:
        msg = str(e).lower()
        assert ("conv" in msg or "cnn" in msg or "compress_cnn" in msg or "shape" in msg or
                "dimension" in msg or "size" in msg or "linear" in msg), \
            f"error is cryptic: {type(e).__name__}: {str(e)[:140]}"

@scenario("B6 already-minimal model [4]")
def t_b6(w):
    r = compress(mk_model(16, hidden=(4,)), mk_data(), **FAST)
    grew_warned = any("LARGER" in str(x.message) for x in w)
    if r.flops_reduction_pct < 0:
        assert r.model_grew, "flops negative but model_grew=False"
        assert grew_warned, "model grew but no UserWarning emitted"

# ── C. API misuse ─────────────────────────────────────────────────────────────

@scenario("C1 target_flops=0.0 rejected with clear ValueError")
def t_c1(w):
    try:
        compress(mk_model(), mk_data(), target_flops=0.0, **FAST)
        raise AssertionError("target_flops=0.0 accepted (degenerate target should be rejected)")
    except ValueError as e:
        assert "target_flops" in str(e)

@scenario("C2 target_flops=1.5 (>1) should be rejected or warned")
def t_c2(w):
    try:
        r = compress(mk_model(), mk_data(), target_flops=1.5, **FAST)
        warned = any("target_flops" in str(x.message).lower() for x in w)
        assert warned, "target_flops=1.5 accepted silently (no validation/warning)"
    except (ValueError, AssertionError) as e:
        if isinstance(e, AssertionError) and "silently" in str(e):
            raise
        pass  # ValueError = good, validated

@scenario("C3 target_flops=-0.5 should be rejected or warned")
def t_c3(w):
    try:
        r = compress(mk_model(), mk_data(), target_flops=-0.5, **FAST)
        warned = any("target_flops" in str(x.message).lower() for x in w)
        assert warned, "target_flops=-0.5 accepted silently (no validation/warning)"
    except (ValueError, AssertionError) as e:
        if isinstance(e, AssertionError) and "silently" in str(e):
            raise
        pass

@scenario("C4 n_generations=0 should be rejected or no-op cleanly")
def t_c4(w):
    try:
        r = compress(mk_model(), mk_data(), n_generations=0, n_pop=3, verbose=False, finetune_epochs=0)
        assert r.model is not None, "returned None model for n_generations=0"
    except ValueError:
        pass

@scenario("C5 n_pop=1")
def t_c5(w):
    r = compress(mk_model(), mk_data(), n_generations=2, n_pop=1, verbose=False, finetune_epochs=0)
    assert r.model is not None

# ── D. Contract consistency ───────────────────────────────────────────────────

@scenario("D1 determinism: same seed twice -> identical arch+acc (docs claim)")
def t_d1(w):
    r1 = compress(mk_model(), mk_data(seed=7), seed=123, **FAST)
    r2 = compress(mk_model(), mk_data(seed=7), seed=123, **FAST)
    assert r1.arch == r2.arch, f"arch differs with same seed: {r1.arch} vs {r2.arch}"
    assert abs(r1.accuracy - r2.accuracy) < 1e-6, f"acc differs with same seed: {r1.accuracy} vs {r2.accuracy}"

@scenario("D2 flops_reduction_pct consistent with original/compressed attrs")
def t_d2(w):
    r = compress(mk_model(), mk_data(), **FAST)
    expected = (r.original_flops - r.compressed_flops) / r.original_flops * 100
    assert abs(r.flops_reduction_pct - expected) < 0.51, \
        f"flops_reduction_pct={r.flops_reduction_pct} but attrs imply {expected:.2f}"

@scenario("D3 compressed_params == sum(p.numel())")
def t_d3(w):
    r = compress(mk_model(), mk_data(), **FAST)
    actual = sum(p.numel() for p in r.model.parameters())
    assert r.compressed_params == actual, f"compressed_params={r.compressed_params} but model has {actual}"

@scenario("D4 model_grew <-> negative reduction coherent")
def t_d4(w):
    r = compress(mk_model(16, hidden=(4,)), mk_data(), **FAST)
    if r.model_grew:
        assert r.flops_reduction_pct < 0, f"model_grew=True but flops_reduction={r.flops_reduction_pct}"
    else:
        assert r.flops_reduction_pct >= 0, f"model_grew=False but flops_reduction={r.flops_reduction_pct}"

@scenario("D5 count_flops exact for known MLP + layers sum to total")
def t_d5(w):
    m = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    total = count_flops(m, input_shape=(512,))
    by_layer = flops_by_layer(m, input_shape=(512,))
    s = sum(by_layer.values())
    assert s == total, f"flops_by_layer sums to {s}, count_flops says {total}"
    handmade_mac = 512*256 + 256*128 + 128*10
    assert total in (handmade_mac, 2*handmade_mac), \
        f"count_flops={total}, expected MACs={handmade_mac} or 2*MACs={2*handmade_mac}"

@scenario("D6 save/load roundtrip -> identical predictions")
def t_d6(w):
    r = compress(mk_model(), mk_data(), **FAST)
    p = os.path.join(TMP, "stress_roundtrip.pt")
    r.save(p)
    r2 = dnaty_load(p)
    x = torch.randn(8, 16)
    r.model.eval(); r2.model.eval()
    with torch.no_grad():
        a, b = r.model(x), r2.model(x)
    assert torch.allclose(a, b, atol=1e-6), f"reloaded model predicts differently (max diff {(a-b).abs().max():.2e})"
    assert r2.arch == r.arch and r2.accuracy == r.accuracy

@scenario("D7 ONNX export numerical parity vs torch")
def t_d7(w):
    try:
        import onnxruntime as ort
    except ImportError:
        raise AssertionError("onnxruntime not installed — parity untested (install to verify)")
    r = compress(mk_model(), mk_data(), **FAST)
    p = os.path.join(TMP, "stress_parity.onnx")
    r.export_onnx(p, input_shape=(16,))
    sess = ort.InferenceSession(p)
    x = torch.randn(4, 16)
    r.model.eval()
    with torch.no_grad():
        t_out = r.model(x).numpy()
    o_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = abs(t_out - o_out).max()
    assert diff < 1e-4, f"ONNX output diverges from torch by {diff:.2e}"

@scenario("D8 benchmark_latency returns sane metrics")
def t_d8(w):
    r = compress(mk_model(), mk_data(), **FAST)
    lat = r.benchmark_latency((16,), n_runs=30)
    for k in ("p50_ms", "p95_ms", "fps"):
        assert k in lat, f"missing key {k} in {list(lat.keys())}"
    assert lat["p50_ms"] > 0 and lat["fps"] > 0
    assert lat["p95_ms"] >= lat["p50_ms"], f"p95 ({lat['p95_ms']}) < p50 ({lat['p50_ms']})"

@scenario("D9 accuracy attr within [0,1]")
def t_d9(w):
    r = compress(mk_model(), mk_data(), **FAST)
    assert 0.0 <= r.accuracy <= 1.0, f"accuracy out of range: {r.accuracy}"

# ── E. Monitoring ─────────────────────────────────────────────────────────────

@scenario("E1 DriftDetector: same dist no drift, shifted dist drifts")
def t_e1(w):
    X = torch.randn(2000, 8)
    det = DriftDetector(threshold=0.2).fit(X)
    same = det.is_drifted(torch.randn(500, 8))
    shifted = det.is_drifted(torch.randn(500, 8) + 3.0)
    assert not same, "flags drift on identically-distributed data"
    assert shifted, "misses an obvious +3 sigma mean shift"

@scenario("E2 DriftDetector constant feature (std=0) no crash")
def t_e2(w):
    X = torch.randn(500, 4); X[:, 2] = 1.0
    det = DriftDetector().fit(X)
    s = det.score(X)
    assert all(v == v for v in s.values() if isinstance(v, float)), f"NaN in drift score: {s}"

@scenario("E3 ProductionTracker predict + meta contract")
def t_e3(w):
    r = compress(mk_model(), mk_data(), **FAST)
    det = DriftDetector().fit(torch.randn(500, 16))
    tr = ProductionTracker(r.model, drift_detector=det)
    preds, meta = tr.predict(torch.randn(12, 16))
    assert preds.shape[0] == 12
    for k in ("psi", "alert"):
        assert k in meta, f"missing meta key {k}: {list(meta.keys())}"

# ── F. CNN paths ──────────────────────────────────────────────────────────────

@scenario("F1 compress_cnn tiny run (early-access claim)")
def t_f1(w):
    from dnaty.core.arch_cnn import DynamicCNN
    X = torch.randn(64, 3, 32, 32); y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)
    r = compress_cnn(DynamicCNN(), loader, n_generations=2, n_pop=2, verbose=False)
    assert r.model is not None

@scenario("F2 compress_with_backbone custom tiny backbone")
def t_f2(w):
    class BB(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                          nn.AdaptiveAvgPool2d(4), nn.Flatten())
            self.classifier = nn.Linear(8*16, 4)
        def forward(self, x):
            return self.classifier(self.features(x))
    X = torch.randn(64, 3, 16, 16); y = torch.randint(0, 4, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)
    r = compress_with_backbone(BB(), loader, n_generations=2, n_pop=3, verbose=False)
    assert r.model is not None

@scenario("F3 prune_conv_channels then forward still works")
def t_f3(w):
    m = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 8, 3, padding=1),
                      nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2))
    before = sum(p.numel() for p in m.parameters())
    pruned = prune_conv_channels(m, amount=0.25)
    out = pruned(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 2)
    after = sum(p.numel() for p in pruned.parameters())
    assert after <= before, f"prune increased params {before}->{after}"

# ── G. load() abuse ───────────────────────────────────────────────────────────

@scenario("G1 load nonexistent file -> clean error")
def t_g1(w):
    try:
        dnaty_load(os.path.join(TMP, "does_not_exist_dnaty.pt"))
        raise AssertionError("load() of missing file returned without error")
    except AssertionError:
        raise
    except FileNotFoundError:
        pass
    except Exception as e:
        raise AssertionError(f"wrong error type for missing file: {type(e).__name__}: {str(e)[:120]}")

@scenario("G2 load corrupted file -> clean error")
def t_g2(w):
    p = os.path.join(TMP, "corrupt_dnaty.pt")
    with open(p, "wb") as f:
        f.write(os.urandom(256))
    try:
        dnaty_load(p)
        raise AssertionError("load() of random bytes returned without error")
    except AssertionError:
        raise
    except Exception:
        pass  # any explicit error is acceptable

# ── run ───────────────────────────────────────────────────────────────────────

ALL = [v for k, v in sorted(globals().items()) if k.startswith("t_") and callable(v)]
print(f"dnaty source version: {dnaty.__version__} | torch {torch.__version__} | scenarios: {len(ALL)}\n")
t0 = time.time()
for fn in ALL:
    fn()
    s, name, dt, msg = RESULTS[-1]
    line = f"[{s:5s}] {name}  ({dt})"
    if msg:
        line += f"\n        -> {msg}"
    print(line, flush=True)

n = {"PASS": 0, "FAIL": 0, "CRASH": 0}
for s, *_ in RESULTS:
    n[s] += 1
print(f"\n==== {n['PASS']} PASS | {n['FAIL']} FAIL | {n['CRASH']} CRASH | total {time.time()-t0:.0f}s ====")
