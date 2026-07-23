"""
DnatyEvolver -- Algorithm 1, optimized for maximum throughput.

Key optimizations:
  1. Sequential local training (CUDA does not benefit from multiple streams)
  2. evaluate() uses torch.inference_mode
  3. Vectorized NSGA-II with numpy
  4. EpisodicMemory with incremental O(1) score updates
  5. model.to(device) is idempotent
"""
from __future__ import annotations
import logging
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

from dnaty.core.individual import Individual
from dnaty.core.arch import DynamicMLP
from dnaty.core.memory import EpisodicMemory, Experience
from dnaty.operators.mutations import OPERATORS, apply_operator
from dnaty.evolution.selection import nsga2_select
from dnaty.training.local_train import local_train, evaluate


class GenerationLog:
    __slots__ = ("gen", "best_acc", "delta_grad", "delta_mem", "op_counts", "n_params")

    def __init__(self, gen, best_acc, delta_grad, delta_mem, op_counts, n_params):
        self.gen = gen
        self.best_acc = best_acc
        self.delta_grad = delta_grad
        self.delta_mem = delta_mem
        self.op_counts = op_counts
        self.n_params = n_params

    def __repr__(self):
        return (f"Gen {self.gen:3d} | acc={self.best_acc:.4f} | "
                f"d_grad={self.delta_grad:.5f} | d_mem={self.delta_mem:.5f} | "
                f"params={self.n_params}")


class DnatyEvolver:
    def __init__(
        self,
        n_pop: int = 20,
        n_generations: int = 50,
        t_local: int = 3,
        lr: float = 1e-3,
        lambda1: float = 1e-4,
        lambda2: float = 1e-6,   # FLOPs weight in Pareto fitness; use 1e-8 for CL
        memory_gamma: float = 0.99,
        memory_tau: float = 1.0,
        top_k_pct: float = 0.03,
        device: str = "cpu",
        input_size: int = 784,
        n_classes: int = 10,
        init_hidden: list[int] | None = None,
        verbose: bool = True,
        n_threads: int | None = None,
        batch_size: int = 512,
        proxy_filter: bool = False,
        proxy_oversample: int = 2,
        warm_start=None,
        warm_start_weight: float = 2.0,
    ):
        self.n_pop = n_pop
        self.n_generations = n_generations
        self.t_local = t_local
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.memory_gamma = memory_gamma
        self.memory_tau = memory_tau
        self.top_k_pct = top_k_pct
        self.device = device
        self.input_size = input_size
        self.n_classes = n_classes
        self.init_hidden = init_hidden or [128, 64]
        self.verbose = verbose
        self.n_threads = n_threads or 1
        self.batch_size = batch_size
        self.proxy_filter = proxy_filter
        self.proxy_oversample = proxy_oversample
        self.history: list[GenerationLog] = []
        self.population: list[Individual] = []
        self.shared_memory = EpisodicMemory(max_size=1000, decay_gamma=memory_gamma)

        # Transferable operator prior (v2.1.0): seed the shared memory so the
        # search starts biased toward operators that worked on a related task.
        self.warm_start_weight = warm_start_weight
        self.n_warm_started = 0
        if warm_start is not None and warm_start_weight != 0:
            self.n_warm_started = self._apply_warm_start(warm_start, warm_start_weight)
            if verbose and self.n_warm_started:
                print(f"[warm_start] seeded {self.n_warm_started} operator priors "
                      f"(weight={warm_start_weight})")

        self._proxy_ensemble = None
        if proxy_filter:
            from dnaty.utils.proxies import ProxyEnsemble
            self._proxy_ensemble = ProxyEnsemble()

    def _apply_warm_start(self, warm_start, weight: float) -> int:
        """Seed self.shared_memory from a prior (dict, path, or EpisodicMemory)."""
        from dnaty.core.memory import load_prior
        if isinstance(warm_start, EpisodicMemory):
            prior = warm_start.to_prior()
        elif isinstance(warm_start, dict):
            prior = warm_start
        elif isinstance(warm_start, (str, bytes)) or hasattr(warm_start, "__fspath__"):
            prior = load_prior(str(warm_start))
        else:
            raise TypeError(
                "warm_start must be a prior dict, a path to a saved prior, or an "
                f"EpisodicMemory instance -- got {type(warm_start).__name__}"
            )
        return self.shared_memory.seed_from_prior(prior, weight=weight)

    def export_prior(self) -> dict:
        """Return the current transferable operator prior (see EpisodicMemory.to_prior)."""
        return self.shared_memory.to_prior()

    def _make_individual(self) -> Individual:
        sizes = [self.input_size] + self.init_hidden
        acts = ["relu"] * len(self.init_hidden)
        model = DynamicMLP(sizes, acts, self.n_classes)
        return Individual(model, EpisodicMemory(decay_gamma=self.memory_gamma))

    def _init_population(self, seed: "Individual | None" = None) -> None:
        if seed is None:
            self.population = [self._make_individual() for _ in range(self.n_pop)]
        else:
            # First individual is the seed; the rest are mutations of the seed
            self.population = [seed.clone()]
            for _ in range(self.n_pop - 1):
                op = np.random.choice(OPERATORS)
                mutant, success = apply_operator(seed, op)
                if not success or not mutant.model.is_valid():
                    mutant = seed.clone()
                self.population.append(mutant)

    def _fitness(self, ind: Individual) -> tuple[float, float, float]:
        # lambda2 controls FLOPs weight in Pareto selection:
        #   1e-6 -> real compression pressure (NAS default)
        #   1e-8 -> negligible FLOPs pressure (CL: preserves accuracy over efficiency)
        cost = ind.count_params() * 1e-5 + ind.count_flops() * self.lambda2
        ind.fitness = (ind.acc, -cost, 0.0)
        return ind.fitness

    def _eval_population(
        self,
        population: list[Individual],
        val_loader: torch.utils.data.DataLoader,
    ) -> list[tuple[float, float, float]]:
        """Evaluate all individuals sequentially using inference_mode."""
        fitnesses = []
        for ind in population:
            acc, _ = evaluate(ind, val_loader, self.device)
            ind.acc = acc
            fitnesses.append(self._fitness(ind))
        return fitnesses

    def _train_parallel(
        self,
        mutated: list[Individual],
        train_loader: torch.utils.data.DataLoader,
    ) -> tuple[list[float], list[float], list[float]]:
        """Train all individuals sequentially -- CUDA is not thread-safe across multiple streams."""
        loss_before, loss_after, grad_norms = [], [], []
        for ind in mutated:
            lb, la, gn = local_train(
                ind, train_loader,
                self.t_local, self.lr,
                self.lambda1, self.lambda2,
                device=self.device,
                batch_size=self.batch_size,
            )
            loss_before.append(lb)
            loss_after.append(la)
            grad_norms.append(gn)
            ind.last_grad_norm = gn
            ind.last_delta_loss = la - lb
        return loss_before, loss_after, grad_norms

    def _mutate_population(self, population: list[Individual]) -> list[Individual]:
        op_probs = self.shared_memory.query_mutation_probs(OPERATORS, tau=self.memory_tau)
        ops   = list(op_probs.keys())
        probs = list(op_probs.values())
        mutated = []
        for ind in population:
            op = np.random.choice(ops, p=probs)
            new_ind, success = apply_operator(ind, op)
            if not success or not new_ind.model.is_valid():
                new_ind = ind.clone()
                new_ind.last_op = "no_op"
            new_ind.parent_acc = ind.acc
            mutated.append(new_ind)
        return mutated

    def _apply_proxy_filter(self, candidates: list[Individual]) -> list[Individual]:
        """
        Zero-cost proxy pre-filter: score all candidates cheaply (no training),
        return the top-n_pop by combined proxy score.

        Halves training cost when proxy_oversample=2: only the top half of
        candidates proceed to local_train + evaluate.
        """
        if self._proxy_ensemble is None or len(candidates) <= self.n_pop:
            return candidates
        input_shape = (self.input_size,)
        scores = []
        for ind in candidates:
            try:
                s = self._proxy_ensemble.score(ind.model, input_shape, batch_size=16)
                scores.append(s["combined"])
            except Exception:
                scores.append(0.0)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [ind for ind, _ in ranked[: self.n_pop]]

    def _update_proxy_weights(
        self,
        score_dicts: list[dict],
        mutated: list[Individual],
        prev_accs: list[float],
    ) -> None:
        """Update proxy ensemble weights based on correlation with actual fitness deltas."""
        if self._proxy_ensemble is None or not score_dicts:
            return
        actual_deltas = [
            ind.acc - (
                ind.parent_acc if ind.parent_acc is not None
                else (prev_accs[i] if i < len(prev_accs) else 0.0)
            )
            for i, ind in enumerate(mutated)
        ]
        self._proxy_ensemble.update(score_dicts, actual_deltas)

    def _update_memory(
        self,
        mutated: list[Individual],
        prev_best_acc: float,
        grad_norms: list[float],
        prev_accs: list[float],  # accuracy of each individual BEFORE mutation
    ) -> float:
        delta_mem = 0.0
        for i, ind in enumerate(mutated):
            if ind.last_op == "no_op":
                continue
            # Improvement relative to the parent (not just the global best).
            # parent_acc travels on the mutant itself because the mutated list
            # may be reordered/oversampled by the proxy filter.
            if ind.parent_acc is not None:
                parent_acc = ind.parent_acc
            else:
                parent_acc = prev_accs[i] if i < len(prev_accs) else prev_best_acc
            if ind.acc > parent_acc + 1e-5:
                exp = Experience(
                    operator=ind.last_op,
                    delta_loss=-(ind.acc - parent_acc),
                    gradient_norm=grad_norms[i],
                    generation=len(self.history),
                )
                self.shared_memory.update(exp)
                delta_mem += exp.impact
        return delta_mem

    def run(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        early_stop_patience: int = 8,
        early_stop_min_delta: float = 1e-4,
        seed_individual: "Individual | None" = None,
        progress_callback=None,
    ) -> tuple[Individual, list[GenerationLog]]:
        self._init_population(seed=seed_individual)

        fitnesses = self._eval_population(self.population, val_loader)
        prev_best_acc = max(ind.acc for ind in self.population)
        best_acc_ever = prev_best_acc
        no_improve_count = 0

        pbar = tqdm(range(1, self.n_generations + 1), disable=not self.verbose)
        for gen in pbar:
            # Phase 1: memory-guided variation
            prev_accs = [ind.acc for ind in self.population]  # parent accuracies

            if self._proxy_ensemble is not None:
                # Oversample mutations, filter cheaply before training
                oversampled = []
                for _ in range(self.proxy_oversample):
                    oversampled.extend(self._mutate_population(self.population))
                proxy_scores = []
                for ind in oversampled:
                    try:
                        proxy_scores.append(
                            self._proxy_ensemble.score(ind.model, (self.input_size,), batch_size=16)
                        )
                    except Exception:
                        proxy_scores.append({"combined": 0.0})
                ranked = sorted(zip(oversampled, proxy_scores), key=lambda x: -x[1]["combined"])
                mutated = [ind for ind, _ in ranked[: self.n_pop]]
                _proxy_score_dicts = [s for _, s in ranked[: self.n_pop]]
            else:
                mutated = self._mutate_population(self.population)
                _proxy_score_dicts = []

            # Phase 2: local training (Lemma 2)
            loss_before_list, loss_after_list, grad_norms = self._train_parallel(
                mutated, train_loader
            )
            delta_grad = float(np.mean([
                b - a for b, a in zip(loss_before_list, loss_after_list)
            ]))

            # Phase 3: multi-objective evaluation
            mut_fitnesses = self._eval_population(mutated, val_loader)

            # Phase 4: NSGA-II selection
            combined_pop = self.population + mutated
            combined_fit = fitnesses + mut_fitnesses
            self.population, fitnesses = nsga2_select(combined_pop, combined_fit, self.n_pop)

            # Phase 5: memory update (Lemma 1)
            delta_mem = self._update_memory(mutated, prev_best_acc, grad_norms, prev_accs)

            # Phase 6: update proxy weights based on correlation with actual fitness
            if _proxy_score_dicts:
                self._update_proxy_weights(_proxy_score_dicts, mutated, prev_accs)

            best_ind = max(self.population, key=lambda ind: ind.acc)
            prev_best_acc = best_ind.acc

            log = GenerationLog(
                gen=gen,
                best_acc=best_ind.acc,
                delta_grad=max(delta_grad, 0.0),
                delta_mem=delta_mem,
                op_counts=dict(self.shared_memory.operator_counts(OPERATORS)),
                n_params=best_ind.count_params(),
            )
            self.history.append(log)

            if self.verbose:
                pbar.set_description(str(log))

            if progress_callback is not None:
                try:
                    progress_callback(log)
                except Exception:
                    logger.warning("progress_callback raised an exception", exc_info=True)

            # -- Early stopping --------------------------------------------
            if best_ind.acc > best_acc_ever + early_stop_min_delta:
                best_acc_ever = best_ind.acc
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    if self.verbose:
                        print(f"\nEarly stop at gen {gen} - no improvement in {early_stop_patience} generations. acc={best_acc_ever:.4f}")
                    break

        best = max(self.population, key=lambda ind: ind.acc)
        return best, self.history


class CnnEvolver(DnatyEvolver):
    """
    Evolver for DynamicCNN -- uses real CNN operators (Conv2D, DepthwiseSep, etc.).
    Same evolution logic as DnatyEvolver, but _make_individual and _mutate_population
    operate on DynamicCNN instead of DynamicMLP.

    Budget-aware operator selection: when the current best individual's FLOPs exceed
    target_flops * baseline_flops, the probability of swap_conv_to_dw and prune_channels
    is boosted by budget_boost_factor so the search prioritises compression.
    """

    def __init__(self, *args, target_flops: float = 0.5, budget_boost_factor: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_flops = target_flops
        self.budget_boost_factor = budget_boost_factor
        self._baseline_flops: float | None = None  # set at run() start

    def _make_individual(self) -> Individual:
        from dnaty.core.arch_cnn import DynamicCNN
        # Use self.n_classes -- critical for datasets that are not CIFAR-10 (10 classes)
        model = DynamicCNN(n_classes=self.n_classes)
        return Individual(model, EpisodicMemory(decay_gamma=self.memory_gamma))

    def _mutate_population(self, population: list[Individual]) -> list[Individual]:
        from dnaty.operators.mutations_cnn import CNN_OPERATORS, apply_cnn_operator
        op_probs = self.shared_memory.query_mutation_probs(CNN_OPERATORS, tau=self.memory_tau)

        # Budget-aware boost: if best individual is still over-budget, triple the
        # probability of compression operators so the search finds cuts faster.
        if self._baseline_flops is not None and population:
            best_flops = max(ind.count_flops() for ind in population)
            budget = self.target_flops * self._baseline_flops
            if best_flops > budget:
                boost_ops = {"swap_conv_to_dw", "prune_channels"}
                raw = {
                    op: (v * self.budget_boost_factor if op in boost_ops else v)
                    for op, v in op_probs.items()
                }
                total = sum(raw.values())
                op_probs = {op: v / total for op, v in raw.items()}

        ops   = list(op_probs.keys())
        probs = list(op_probs.values())
        mutated = []
        for ind in population:
            op = np.random.choice(ops, p=probs)
            new_ind, success = apply_cnn_operator(ind, op)
            if not success or not new_ind.model.is_valid():
                new_ind = ind.clone()
                new_ind.last_op = "no_op"
            new_ind.parent_acc = ind.acc
            mutated.append(new_ind)
        return mutated

    def _init_population(self, seed=None):
        super()._init_population(seed=seed)
        # Capture baseline FLOPs right after initial population is built so budget
        # checks in _mutate_population have a stable reference throughout evolution.
        self._baseline_flops = float(np.mean([ind.count_flops() for ind in self.population]))


class QuantAwareEvolver(DnatyEvolver):
    """
    Quantization-aware NAS: fitness evaluation uses dynamic INT8 quantization.

    Before computing accuracy for each individual, applies
    torch.quantization.quantize_dynamic (INT8 for all nn.Linear layers).
    This ensures the selected architecture remains accurate after quantization
    — architectures that degrade under int8 are naturally filtered out.

    Works on CPU without a calibration dataset. Adds ~20% overhead per
    generation vs standard DnatyEvolver.

    Args:
        dtype: quantization dtype (default torch.qint8).
    """

    def __init__(self, *args, dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._quant_dtype = dtype or torch.qint8

    def _eval_population(self, population, val_loader):
        fitnesses = []
        for ind in population:
            # Quantize a temporary copy, evaluate, discard — keep float weights
            import copy
            q_model = torch.quantization.quantize_dynamic(
                copy.deepcopy(ind.model).cpu(),
                {torch.nn.Linear},
                dtype=self._quant_dtype,
            )
            # Temporarily swap model for eval
            orig_model = ind.model
            ind.model = q_model
            acc, _ = evaluate(ind, val_loader, "cpu")
            ind.model = orig_model
            ind.acc = acc
            fitnesses.append(self._fitness(ind))
        return fitnesses


class LatencyEvolver(DnatyEvolver):
    """
    Hardware-aware NAS: minimises (1 - accuracy, latency_ms) instead of FLOPs.

    Replaces the FLOPs proxy with real ONNX Runtime latency measured on the
    current CPU (or estimated for a target device via hw_detect scaling tables).

    Fitness: (accuracy, -latency_ms)  → NSGA-II maximises both.

    Args:
        target_device: "cpu" measures on the current machine; "rpi4", "rpi5",
                       etc. scale the measured latency via hw_detect tables.
        latency_weight: weight of latency vs accuracy in Pareto fitness cost.
        predictor_path: optional path to a pre-trained GBM surrogate. When
                        provided, the predictor is used for 80% of evaluations
                        and real measurement is used for the remaining 20%.
        measure_every: measure real latency every N generations (others use
                       predictor). Ignored if no predictor is loaded.
    """

    def __init__(
        self,
        *args,
        target_device: str = "cpu",
        latency_weight: float = 0.01,
        predictor_path: str | None = None,
        measure_every: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_device = target_device
        self.latency_weight = latency_weight
        self.measure_every = measure_every
        self._gen_counter = 0

        from dnaty.utils.hw_detect import latency_scale, detect_hw
        self._scale = latency_scale(target_device)
        hw = detect_hw()
        if self.verbose:
            print(f"[LatencyEvolver] device={hw['device_class']}  "
                  f"target={target_device}  scale={self._scale:.1f}x")

        from dnaty.utils.latency_predictor import LatencyPredictor
        self._predictor = LatencyPredictor(model_path=predictor_path)
        if self._predictor.is_trained() and self.verbose:
            print(f"[LatencyEvolver] GBM surrogate loaded — 80% predictions, 20% measured")
        elif self.verbose:
            print(f"[LatencyEvolver] No surrogate — measuring latency every generation")

    def _measure_latency_ms(self, ind: "Individual") -> float:
        """
        Latency for one individual.

        Priority order:
          1. Lookup table  (µs-level, zero overhead, ~80% hit rate)
          2. GBM surrogate (if trained)
          3. ONNX Runtime  (actual measurement, ~30 runs)
          4. Analytical    (params × constant, last resort)
        """
        # 1. Lookup table: fast path for common MLP patterns
        try:
            from dnaty.utils.latency_tables import estimate_mlp_latency
            sizes = list(getattr(ind.model, "layer_sizes", []))
            if len(sizes) >= 2:
                n_classes = getattr(ind.model, "n_classes", self.n_classes)
                # layer_sizes never includes the classifier — always append it
                # (the old `if sizes[-1] != n_classes` skipped a real layer
                # whenever the last hidden width coincided with n_classes).
                full_sizes = sizes + [n_classes]
                table_ms = estimate_mlp_latency(full_sizes, device=self.target_device)
                if table_ms > 0:
                    return table_ms
        except Exception:
            pass

        # 2. GBM surrogate (handled by caller via _get_latency_ms)
        # 3. ONNX Runtime measurement
        from dnaty.utils.latency_bench import measure_latency
        try:
            result = measure_latency(ind.model, input_shape=(self.input_size,),
                                     n_warmup=5, n_runs=30)
            return result["p50_ms"] * self._scale
        except Exception:
            # 4. Analytical fallback
            return ind.count_params() * 1e-5 * self._scale

    def _predict_latency_ms(self, ind: "Individual") -> float:
        """GBM surrogate prediction × device scale."""
        sizes = getattr(ind.model, 'layer_sizes', [])
        # ALL hidden widths (layer_sizes = [input, h1, ..., hN]; classifier is
        # separate) — must match arch_features() in scripts/build_latency_dataset.py,
        # which trains the GBM on every Linear out_features except the classifier.
        widths = [s for s in sizes[1:]] if len(sizes) > 1 else []
        feats = {
            "n_layers": len(widths),
            "widths": widths or [128],
            "total_params": ind.count_params(),
            "total_flops": ind.count_flops(),
            "input_size": self.input_size,
        }
        return self._predictor.predict_ms(feats) * self._scale

    def _get_latency_ms(self, ind: "Individual", use_surrogate: bool) -> float:
        if use_surrogate and self._predictor.is_trained():
            return self._predict_latency_ms(ind)
        return self._measure_latency_ms(ind)

    def _fitness(self, ind: "Individual") -> tuple[float, float, float]:
        use_surrogate = (self._predictor.is_trained() and
                         self._gen_counter % self.measure_every != 0)
        latency_ms = self._get_latency_ms(ind, use_surrogate)
        ind.latency_ms = latency_ms
        cost = latency_ms * self.latency_weight
        ind.fitness = (ind.acc, -cost, 0.0)
        return ind.fitness

    def run(self, train_loader, val_loader, **kwargs):
        self._gen_counter = 0
        original_verbose = self.verbose

        class _CountingEvolver:
            pass

        # Intercept each generation to increment counter
        _orig_update = self._update_memory

        def _counting_update_memory(*args, **kw):
            self._gen_counter += 1
            return _orig_update(*args, **kw)

        self._update_memory = _counting_update_memory
        try:
            result = super().run(train_loader, val_loader, **kwargs)
        finally:
            self._update_memory = _orig_update
        return result

    def pareto_front(self) -> list[dict]:
        """
        Returns the current Pareto front as a list of dicts with
        accuracy, latency_ms, params for inspection / export.
        """
        front = []
        for ind in self.population:
            front.append({
                "accuracy": round(ind.acc, 4),
                "latency_ms": round(getattr(ind, "latency_ms", 0.0), 3),
                "params": ind.count_params(),
                "flops": ind.count_flops(),
            })
        return sorted(front, key=lambda x: -x["accuracy"])


class QuantLatencyEvolver(QuantAwareEvolver, LatencyEvolver):
    """
    INT8 fitness + latency objective — used by compress(target="latency",
    quant_aware=True).

    MRO: accuracy is evaluated on a temporarily-quantized copy
    (QuantAwareEvolver._eval_population), while fitness cost comes from
    measured/estimated latency (LatencyEvolver._fitness). Constructor kwargs of
    both parents (dtype, target_device, latency_weight, ...) are accepted.
    """
