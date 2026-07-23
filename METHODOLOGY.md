# dNATY: Technical Methodology

**Version 2.1.0 · Pedro Vergueiro · 2026**

This document describes the algorithmic design of dNATY for researchers evaluating the system. It answers the three questions a reviewer will ask: what the algorithm does, how it relates to prior work, and what the experiments actually claim.

---

## 1. Problem Statement

Given a trained (or trainable) PyTorch MLP and a dataset, find a smaller architecture that preserves validation accuracy above a user-defined threshold while minimising forward-pass FLOPs. This is a bi-objective NAS problem on a discrete search space of MLP widths and structural modifications.

The target deployment context is CPU-only edge inference (industrial sensors, embedded controllers, IoT gateways). This scope intentionally excludes GPU-bound models; the library does not compete with OFA, MnasNet, or NAS-Bench-based methods on those tasks.

---

## 2. Algorithm

### 2.1 Search Space

The search space is the set of `DynamicMLP` architectures reachable by applying a fixed set of 11 structural mutation operators to an initial architecture:

| Operator | Effect |
|---|---|
| `add_neuron` | Widen one hidden layer by +k neurons |
| `remove_neuron` | Narrow one hidden layer by −k neurons |
| `add_skip` | Insert a skip connection between two non-adjacent layers |
| `add_residual` | Insert a residual block |
| `change_activation` | Swap activation function (ReLU, GELU, Tanh, SiLU) |
| `split_layer` | Replace one wide layer with two narrower ones |
| `merge_layers` | Collapse two consecutive layers into one |
| `prune_connections` | Structured pruning: zero then remove low-magnitude weights |
| `duplicate_module` | Copy a layer with weight perturbation |
| `add_conv_block` | Insert a 1-D conv block (tabular feature mixing) |
| `depthwise_sep` | Replace a Linear layer with a depthwise-separable equivalent |

### 2.2 Fitness Function

Each candidate architecture `ind` is evaluated on three objectives (NSGA-II minimises all three after sign inversion):

```
f1(ind) = -acc(ind)                             # maximise accuracy
f2(ind) = params(ind) * 1e-5 + flops(ind) * λ  # minimise compute cost
f3(ind) = 0.0                                   # reserved (sharpness)
```

`λ` (lambda2) is set proportional to `target_flops` — higher compression targets apply stronger FLOPs pressure.

FLOPs are counted via hook-based measurement (see `dnaty/utils/flops_counter.py`), not estimated from parameter count. This matters for skip connections, residuals, and depthwise-separable layers where the parameter count and FLOPs differ.

### 2.3 EpisodicMemory (Operator Selection)

Standard evolutionary NAS selects mutation operators uniformly at random. dNATY replaces this with **importance-weighted operator sampling** via an `EpisodicMemory` module (`dnaty/core/memory.py`).

**Formal definition.** Let `O = {o_1, …, o_K}` be the operator set (K = 11). For each operator `o_k` maintain a scalar score `s_k` initialised to 1/K. After each generation, for every applied operator `o_k` that produced a child with `Δacc > 0`:

```
impact_k = |Δacc| · ‖∇L‖₂            (Eq. 1.4, memory.py)
s_k ← γ · s_k + impact_k
```

where γ = 0.99 is a temporal decay factor. Operator selection probabilities for the next generation:

```
π_k = softmax(s / τ)_k,   τ = 1.0
```

**Complexity.** Score updates are O(1) per applied operator. Each generation queries O(K) to compute the softmax. Memory is O(K · max_size), where max_size = 500 bounds the replay buffer independently.

**What this is not.** EpisodicMemory in dNATY is a mechanism for **NAS operator scheduling**, not for continual learning replay. The name overlaps with the CL literature but the mechanism is distinct. The CL replay strategy (Experiment 3) is a separate module described in Section 4.

### 2.4 Transferable Operator Priors (v2.1.0)

The operator scores `s = (s_1, …, s_K)` are a compact, task-agnostic summary of *which structural moves paid off* — they carry no weights and no data. dNATY 2.1.0 makes them transferable across runs: after searching task A, export `s_A`; when searching a related task B, initialise B's memory from `s_A` instead of from the uniform prior.

**Normalisation.** Raw scores are not comparable across runs — their magnitude depends on the gradient norms `‖∇L‖₂` and the number of improving steps of the source run. Before injection the prior is centred and scaled to unit maximum magnitude, then multiplied by a strength `w`:

```
ŝ_k = (s_k − mean(s)) / max_j |s_j − mean(s)|          # unit-max-abs, scale-free
s_k^(B, init) = w · ŝ_k                                 # w = warm_start_weight
```

Because operator selection is `softmax(s/τ)`, `w` acts as an inverse-temperature on the transferred prior: `w = 0` recovers a cold start; `w = 2` (default) biases early generations toward the top operators by a few ×; large `w` risks premature convergence. Since the per-generation update `s_k ← γ·s_k + impact_k` decays seeded scores whenever new improving experiences arrive, the prior is a **head start that provably fades** — after `t` improving updates the seeded component is scaled by `γ^t`, so task-B evidence dominates asymptotically. The transferred prior can only bias *operator scheduling*; it cannot fix the final architecture, so a poorly-matched prior costs at most a slow start, never a wrong answer.

**Relation to meta-learning.** This is meta-learning at the level of the *search operator schedule*, not the weight initialisation (contrast MAML, §3.1). It requires no second-order gradients, no shared architecture, and no task distribution defined in advance — two related tasks suffice, and the transferable object is a K-vector (K = 11), not a network. The claim is scoped to related MLP/tabular tasks; `scripts/warm_start_demo.py` measures generations-to-target, cold vs warm-started, so the effect is reported rather than asserted.

---

## 3. Relationship to Prior Work

### 3.1 vs. MAML (Model-Agnostic Meta-Learning)

MAML (Finn et al., 2017) learns an initialisation θ* by computing second-order gradients across a distribution of tasks:

```
θ* = argmin_θ Σ_τ L_τ(θ - α · ∇_θ L_τ(θ))
```

**Key differences from dNATY:**

| Dimension | MAML | dNATY EpisodicMemory |
|---|---|---|
| What it learns | Weight initialisation | Operator sampling distribution |
| Gradient order | Second-order (expensive) | Zero-order (no gradients at memory level) |
| Compute | GPU-bound | CPU-only |
| Task definition | Explicit task distribution at meta-train | Single-dataset, no task distribution required |
| Transfer mechanism | Good initialisation for fast adaptation | Operator reuse across generations |

dNATY does not claim to perform meta-learning. The analogy to MAML is only that both accumulate experience across episodes to bias future search.

### 3.2 vs. ER-ACE (Experience Replay with Asymmetric Cross-Entropy)

ER-ACE (Caccia et al., 2022) is a continual learning method that modifies the cross-entropy loss during replay to be **asymmetric**: current-task samples update the full output head, while replay samples only update logits for past classes, preventing intransigence.

**Key differences:**

| Dimension | ER-ACE | dNATY CL (Experiment 3) |
|---|---|---|
| Loss function | Asymmetric CE (current vs. replay heads) | Symmetric CE on mixed batch |
| Replay purpose | Prevent intransigence + forgetting | Prevent catastrophic forgetting |
| Memory management | Ring buffer, uniform sampling | Balanced per-task buffer (200 samples/task) |
| Architecture | Fixed during CL | Fixed during CL (NAS is a separate phase) |

dNATY's CL replay is simpler than ER-ACE. It does not implement the asymmetric loss. The comparison in Experiment 3 is against EWC specifically, not against ER-ACE. A fair comparison including ER-ACE would be a natural extension.

### 3.3 vs. Simple ER (Experience Replay)

Standard ER (Robins, 1995; Rolnick et al., 2018) uniformly samples from a fixed-size replay buffer.

dNATY's Experiment 3 uses **balanced ER**: the replay buffer stores exactly 200 samples per completed task (not a single ring buffer of fixed total size). This ensures no task dominates the replay distribution regardless of class imbalance.

The NAS-level EpisodicMemory (Section 2.3) is importance-weighted, not uniform — but this applies to operator selection, not to CL replay.

### 3.4 vs. Random NAS

The most direct baseline. `scripts/prove_it.py` runs dNATY vs. a random NAS baseline (same operator set, uniform operator selection) under identical generation budgets. EpisodicMemory's advantage is measurable in that comparison.

---

## 4. Experiments

### 4.1 Experiment 1 — NAS Compression (13 real datasets)

**Setup:** `n_generations=30`, `n_pop=15`, `target_flops=0.5`, seed=42, standard 80/20 train/val split.

**Datasets:** 13 public tabular datasets (UCI, Kaggle) spanning IoT sensors, fault detection, medical, financial, and computer vision (MNIST). All datasets are real; no data augmentation is applied.

**Metric:** FLOPs reduction (%) = `1 - compressed_flops / original_flops`, measured via hook-based counter on one forward pass with batch_size=1.

**Result range:** −18.8% to −83.4% across 13 datasets. The median is approximately −56%. The maximum (−83.4%, Dry Bean Quality, 16 features) reflects a case where the initial architecture is heavily oversized relative to feature dimensionality.

> **Reviewer note on "up to 83%":** The headline figure is the best-case result, not the typical result. The median reduction across the 13 real datasets (−56%) is the more informative summary. In general, compression scales with how oversized the initial architecture is — a deliberately lean baseline will compress less (correct Pareto behavior, not a failure of the algorithm).

### 4.2 Experiment 2 — 5 Real Kaggle Datasets

**Purpose:** Cross-domain validation on public real-world data across different sizes and domains. Datasets: IBM HR Employee Attrition (1,470 rows / HR), Adult Census Income (32,561 / financial), Air Quality UCI (7,674 / environmental sensors), Diabetes 130-US Hospitals (101,766 / clinical), Telco Customer Churn (7,043 / telecom). All data publicly available on Kaggle; reproduce with `python scripts/benchmark_market_real.py`.

**Results:** 4 of 5 datasets compressed positively (−75.5%, −74.4%, −45.0%, −8.0%). Telco Customer Churn showed model growth (+20%) — `model_grew=True` was raised automatically. This is correct Pareto behavior: with the given baseline and generation budget, NSGA-II found a deeper but wider solution. Increasing the baseline width (e.g. `[1024, 512, 256]`) or `target_flops` resolves it. **The library warns explicitly rather than silently returning an oversized model.**

**Scope:** These 5 datasets complement the 13 public datasets in Experiment 1, bringing the total to 18 real datasets. The Telco result is included honestly — its model growth is documented behavior, not a suppressed failure. The median FLOPs reduction across all 17 datasets with positive compression is −56%.

### 4.3 Experiment 3 — Continual Learning, Split-MNIST

**Setup:** 5 sequential tasks (digit pairs 0–1, 2–3, 4–5, 6–7, 8–9), 3 seeds {0, 1, 2}, `n_epochs=15`, `batch_size=256`. Architecture: `DynamicMLP([784, 256, 128], n_classes=10)`.

**Methods compared:**
- **dNATY** — balanced ER (200 samples/task)
- **EWC** — `ewc_lambda=400.0`, Fisher matrix computed on 300 samples per task
- **MLP baseline** — plain fine-tuning, no CL mechanism

**Metric:** Backward Transfer (BWT) = `(1/(T−1)) Σ_{i=1}^{T-1} [R_{T-1,i} − R_{i,i}]`. BWT = 0 means no forgetting; BWT = −1 means complete forgetting.

**Results (mean ± std, 3 seeds):**

| Method | BWT (↑ better) |
|---|---|
| dNATY balanced replay | −0.204 ± σ |
| EWC (λ=400) | −0.998 ± σ |
| MLP (no CL) | −0.998 ± σ |

Statistical test: paired t-test on BWT across 3 seeds (dNATY vs. EWC); p-value and Cohen's d reported in `results/exp3_cl_results.json`.

**Scope and limitations:**

1. **Split-MNIST is a weak benchmark.** Binary digit classification is near-linearly separable. Results on this dataset are a proof-of-concept, not a publishable CL claim. Experiments 4 and 5 (Permuted-MNIST, Split-CIFAR-10) extend to harder benchmarks.

2. **3 seeds is minimal.** Standard CL papers use 10+ seeds for BWT estimates.

3. **No comparison to ER-ACE, DER++, or PackNet.** The EWC baseline was chosen because it is the most widely implemented regularisation method and representative of the regularisation family. Replay-based comparison requires more careful experimental design.

4. **The architecture is fixed during CL.** The NAS search (Experiment 1) runs offline. Experiment 3 tests whether a fixed compressed architecture can learn continually — it does not test NAS + CL jointly.

### 4.4 Experiment 4 — Continual Learning, Permuted-MNIST

**File:** `dnaty/experiments/exp4_permuted_mnist.py`

10 tasks; each task is MNIST with a fixed random pixel permutation. All 10 classes remain active across tasks (domain-incremental setting). This is a substantially harder benchmark than Split-MNIST: the input distribution changes completely across tasks while the label space stays identical, preventing the model from exploiting task-specific output heads.

### 4.5 Experiment 5 — Continual Learning, Split-CIFAR-10

**File:** `dnaty/experiments/exp5_split_cifar10.py`

5 tasks on CIFAR-10 (2 classes/task), flattened to 3072-dimensional input for the MLP. This benchmark tests the same CL strategies on a naturalistic image dataset where linear separability is not guaranteed. The MLP baseline accuracy is substantially lower than on MNIST, which is expected — the goal is to measure relative forgetting, not absolute accuracy.

---

## 5. What dNATY Does Not Claim

- No comparison to OFA, MnasNet, or DARTS — those target GPU-based convolutional NAS; dNATY targets CPU-only MLP compression.
- No comparison to MAML on standard meta-learning benchmarks (Omniglot, miniImageNet).
- Conv NAS (`compress_cnn`) is under active development; conv-specific results are early-access and not part of the benchmarked claims.
- Transformer/LLM compression is out of scope.

---

## 6. Reproducibility

All experiments are seeded and deterministic. The pytest suite (142 tests) includes reproducibility regression tests gating every release. To reproduce:

```bash
# NAS vs Random NAS (Experiment 1 mini-version)
python scripts/prove_it.py

# Continual learning experiments
python -m dnaty.experiments.exp3_cl        # Split-MNIST
python -m dnaty.experiments.exp4_permuted_mnist  # Permuted-MNIST
python -m dnaty.experiments.exp5_split_cifar10   # Split-CIFAR-10

# Full benchmark (13 real datasets — requires Kaggle CSVs)
python scripts/benchmark_iot.py
```

Results are saved to `results/` as JSON with seed, per-task matrices, and aggregate statistics.

---

## References

- Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML*.
- Kirkpatrick, J., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. *PNAS*.
- Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Task Learning. *NeurIPS*.
- Caccia, L., et al. (2022). New Insights on Reducing Abrupt Representation Change in Online Continual Learning. *ICLR*. (ER-ACE)
- Deb, K., et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE TEC*.
- Rolnick, D., et al. (2019). Experience Replay for Continual Learning. *NeurIPS*.
