"""
v2.2.0 — "Lifelong Edge" feature tests.

Covers the four features of the release:
  1. Plasticity preservation (utility tracking, dormant reinit, effective rank)
  2. Lifelong adaptation API (ReplayBuffer + CompressResult.adapt)
  3. Meta-learned search controller (LinUCB + evolver integration)
  4. Federated prior merging (merge_priors)
"""
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from dnaty.core.arch import DynamicMLP
from dnaty.core.memory import EpisodicMemory, merge_priors
from dnaty.evolution.controller import MetaController, CONTEXT_DIM
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.result import CompressResult
from dnaty.training.plasticity import (
    NeuronUtilityTracker,
    PlasticityController,
    effective_rank,
    reinit_dormant,
)
from dnaty.training.replay import ReplayBuffer


def _mlp(sizes=(16, 12, 8), n_classes=4) -> DynamicMLP:
    return DynamicMLP(list(sizes), n_classes=n_classes)


def _result_for(model: DynamicMLP) -> CompressResult:
    p = model.count_params()
    f = model.count_flops()
    return CompressResult(
        model=model,
        original_flops=f * 2, compressed_flops=f,
        original_params=p * 2, compressed_params=p,
        accuracy=0.9, flops_reduction=0.5, generations=1,
        arch=list(model.layer_sizes[1:]),
    )


# ---------------------------------------------------------------------------
# 1. Plasticity
# ---------------------------------------------------------------------------

class TestPlasticity:
    def test_utilities_shape_and_nonneg(self):
        model = _mlp()
        tracker = NeuronUtilityTracker(model)
        tracker.update(torch.randn(32, 16))
        utils = tracker.utilities()
        assert [u.numel() for u in utils] == [12, 8]
        assert all((u >= 0).all() for u in utils)
        assert tracker.n_observed == 1

    def test_dormant_mask_selects_lowest(self):
        model = _mlp()
        tracker = NeuronUtilityTracker(model)
        tracker._util = [torch.arange(12.0), torch.arange(8.0)]
        masks = tracker.dormant_mask(0.25)
        # 25% of 12 = 3 lowest; 25% of 8 = 2 lowest
        assert masks[0].sum().item() == 3 and masks[0][:3].all()
        assert masks[1].sum().item() == 2 and masks[1][:2].all()

    def test_dormant_mask_never_kills_whole_layer(self):
        model = _mlp((4, 2), n_classes=3)
        tracker = NeuronUtilityTracker(model)
        tracker.update(torch.randn(8, 4))
        masks = tracker.dormant_mask(0.99)
        assert masks[0].sum().item() <= 1  # at least one unit survives

    def test_reinit_equals_ablation_no_random_noise(self):
        # The reborn unit must be silent: the post-reinit model computes
        # exactly what the pre-reinit model computes with the dormant unit
        # ablated. None of the fresh random weights may leak into the output.
        torch.manual_seed(0)
        model = _mlp((16, 12, 8))
        model.eval()
        x = torch.randn(64, 16)
        masks = [torch.zeros(12, dtype=torch.bool), torch.zeros(8, dtype=torch.bool)]
        masks[0][[1, 5]] = True
        masks[1][[0]] = True
        # Reference: ablate the same units (zero their outgoing columns only).
        import copy
        ablated = copy.deepcopy(model)
        with torch.no_grad():
            ablated.net[3].weight[:, [1, 5]] = 0.0   # consumer of hidden 0
            ablated.net[-1].weight[:, [0]] = 0.0     # classifier consumes hidden 1
        expected = ablated(x)
        n = reinit_dormant(model, masks, seed=1)
        assert n == 3
        after = model(x)
        assert torch.allclose(expected, after, atol=1e-6), \
            "reinit output differs from ablation — fresh weights leaked"

    def test_reinit_gives_fresh_incoming_weights(self):
        torch.manual_seed(0)
        model = _mlp()
        lin0 = model.net[0]
        old_row = lin0.weight[3].clone()
        masks = [torch.zeros(12, dtype=torch.bool), torch.zeros(8, dtype=torch.bool)]
        masks[0][3] = True
        reinit_dormant(model, masks, seed=2)
        assert not torch.allclose(lin0.weight[3], old_row)
        assert lin0.bias[3].item() == 0.0
        # BN channel reset
        bn0 = model.net[1]
        assert bn0.running_mean[3].item() == 0.0
        assert bn0.running_var[3].item() == 1.0

    def test_reinit_zeroes_skip_projection_columns(self):
        model = _mlp((16, 12, 8))
        proj = nn.Linear(12, 8, bias=False)
        model.add_skip_connection(1, 2, proj)  # src=1 -> hidden layer 0
        masks = [torch.zeros(12, dtype=torch.bool), torch.zeros(8, dtype=torch.bool)]
        masks[0][7] = True
        reinit_dormant(model, masks)
        assert (proj.weight[:, 7] == 0).all()

    def test_effective_rank_bounds_and_collapse(self):
        model = _mlp((16, 12, 8))
        model.eval()
        r = effective_rank(model, torch.randn(64, 16))
        assert 1.0 <= r <= 8.0 + 1e-6
        # Rank-collapsed input (all rows identical) -> features identical ->
        # centred matrix is 0 -> effective rank degenerates to 1.
        x1 = torch.randn(1, 16).repeat(64, 1)
        assert effective_rank(model, x1) == pytest.approx(1.0)

    def test_controller_fires_on_schedule(self):
        model = _mlp()
        ctrl = PlasticityController(model, reinit_fraction=0.25,
                                    check_every=3, min_observed=3, seed=0)
        x = torch.randn(16, 16)
        assert ctrl.maybe_reinit() == 0  # nothing observed yet
        for _ in range(3):
            ctrl.observe(x)
        n = ctrl.maybe_reinit()
        assert n > 0
        assert ctrl.n_firings == 1
        assert ctrl.total_reborn == n
        assert ctrl.maybe_reinit() == 0  # schedule reset
        assert len(ctrl.rank_history) == 1

    def test_model_learns_after_reinit(self):
        torch.manual_seed(0)
        model = _mlp((8, 16), n_classes=2)
        x = torch.randn(128, 8)
        y = (x[:, 0] > 0).long()
        masks = [torch.zeros(16, dtype=torch.bool)]
        masks[0][:4] = True
        reinit_dormant(model, masks, seed=0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        crit = nn.CrossEntropyLoss()
        model.train()
        first = crit(model(x), y).item()
        for _ in range(30):
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        assert loss.item() < first, "model failed to train after reinit"


# ---------------------------------------------------------------------------
# 2. Replay buffer + adapt
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_capacity_bound_and_seen(self):
        buf = ReplayBuffer(capacity=10, seed=0)
        for _ in range(5):
            buf.add_batch(torch.randn(7, 4), torch.zeros(7, dtype=torch.long))
        assert len(buf) == 10
        assert buf.seen == 35

    def test_keeps_everything_under_capacity(self):
        buf = ReplayBuffer(capacity=100, seed=0)
        x = torch.arange(12.0).reshape(6, 2)
        y = torch.arange(6)
        buf.add_batch(x, y)
        sx, sy = buf.sample(6)
        assert sorted(sy.tolist()) == list(range(6))

    def test_sample_shapes_and_no_replacement(self):
        buf = ReplayBuffer(capacity=8, seed=1)
        buf.add_batch(torch.randn(8, 3), torch.arange(8))
        sx, sy = buf.sample(5)
        assert sx.shape == (5, 3) and sy.shape == (5,)
        assert len(set(sy.tolist())) == 5
        # Requesting more than stored caps at the stored size
        sx2, _ = buf.sample(50)
        assert sx2.shape[0] == 8

    def test_empty_sample_raises(self):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=4).sample(1)

    def test_state_dict_roundtrip(self):
        buf = ReplayBuffer(capacity=6, seed=0)
        buf.add_batch(torch.randn(9, 2), torch.arange(9))
        state = buf.state_dict()
        buf2 = ReplayBuffer(capacity=1)
        buf2.load_state_dict(state)
        assert len(buf2) == len(buf) == 6
        assert buf2.seen == 9
        sx, sy = buf2.sample(6)
        assert sx.shape == (6, 2)


class TestAdapt:
    def _drift_data(self, n=256, seed=0, shift=0.0):
        rng = np.random.default_rng(seed)
        x = rng.normal(shift, 1.0, size=(n, 8)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] > 2 * shift).astype(np.int64)
        return x, y

    def test_adapt_improves_on_new_distribution(self):
        torch.manual_seed(0)
        model = _mlp((8, 32, 16), n_classes=2)
        result = _result_for(model)
        x, y = self._drift_data(shift=1.5)
        m = result.adapt(x, y, epochs=8, lr=1e-3, seed=0)
        assert m["acc_after"] > m["acc_before"]
        assert m["replay_size"] > 0
        assert m["retention_before"] is None  # first call: no past yet
        assert result.adapt_history == [m]

    def test_adapt_measures_retention_on_second_call(self):
        torch.manual_seed(0)
        model = _mlp((8, 32, 16), n_classes=2)
        result = _result_for(model)
        x0, y0 = self._drift_data(seed=0, shift=0.0)
        result.adapt(x0, y0, epochs=8, lr=1e-3, seed=0)
        x1, y1 = self._drift_data(seed=1, shift=2.0)
        m = result.adapt(x1, y1, epochs=8, lr=1e-3, seed=0)
        assert m["retention_before"] is not None
        assert m["retention_after"] is not None
        # Replay keeps the old distribution from being wiped out entirely.
        assert m["retention_after"] >= 0.5

    def test_adapt_plasticity_off_reborn_zero(self):
        model = _mlp((8, 16), n_classes=2)
        result = _result_for(model)
        x, y = self._drift_data()
        m = result.adapt(x, y, epochs=1, plasticity=False, seed=0)
        assert m["n_reborn"] == 0

    def test_adapt_plasticity_on_reborn_positive(self):
        model = _mlp((8, 32, 32), n_classes=2)
        result = _result_for(model)
        x, y = self._drift_data()
        m = result.adapt(x, y, epochs=1, reinit_fraction=0.1, seed=0)
        assert m["n_reborn"] > 0

    def test_adapt_validates_input(self):
        result = _result_for(_mlp((8, 16), n_classes=2))
        with pytest.raises(ValueError):
            result.adapt(np.zeros((3, 8)), np.zeros(2))
        with pytest.raises(ValueError):
            result.adapt(np.zeros((0, 8)), np.zeros(0))


# ---------------------------------------------------------------------------
# 3. Meta-learned controller
# ---------------------------------------------------------------------------

class TestMetaController:
    def _ctx(self, **kw):
        base = dict(gen=5, n_generations=30, best_acc=0.8,
                    param_ratio=1.0, delta_grad=0.1, no_improve=1)
        base.update(kw)
        return MetaController.make_context(**base)

    def test_context_shape_and_bounds(self):
        ctx = self._ctx(param_ratio=100.0, delta_grad=50.0, no_improve=99)
        assert ctx.shape == (CONTEXT_DIM,)
        assert ctx[0] == 1.0
        assert np.all(np.abs(ctx) <= 1.0 + 1e-9)

    def test_update_shifts_probs_toward_rewarded_op(self):
        ctrl = MetaController(alpha=0.0, warmup=1)
        ops = ["op_a", "op_b", "op_c"]
        ctx = self._ctx()
        for _ in range(30):
            ctrl.update("op_a", ctx, reward=0.05)
            ctrl.update("op_b", ctx, reward=-0.05)
            ctrl.update("op_c", ctx, reward=0.0)
        probs = ctrl.select_probs(ops, ctx)
        assert probs["op_a"] > probs["op_c"] > probs["op_b"]
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_context_dependence(self):
        # op_a helps early, op_b helps late — the controller must learn the flip.
        ctrl = MetaController(alpha=0.0, warmup=1)
        early = self._ctx(gen=1)
        late = self._ctx(gen=30)
        for _ in range(40):
            ctrl.update("op_a", early, 0.05)
            ctrl.update("op_b", early, -0.05)
            ctrl.update("op_a", late, -0.05)
            ctrl.update("op_b", late, 0.05)
        assert ctrl.select_probs(["op_a", "op_b"], early)["op_a"] > 0.5
        assert ctrl.select_probs(["op_a", "op_b"], late)["op_b"] > 0.5

    def test_trust_grows_and_blend_cold_start_is_memory(self):
        ctrl = MetaController(warmup=10)
        assert ctrl.trust == 0.0
        mem = {"op_a": 0.7, "op_b": 0.3}
        blended = ctrl.blend(mem, self._ctx())
        assert blended == pytest.approx(mem)
        for _ in range(10):
            ctrl.update("op_a", self._ctx(), 0.01)
        assert ctrl.trust == pytest.approx(0.5)

    def test_blend_restricted_to_memory_ops(self):
        ctrl = MetaController(warmup=1)
        ctrl.update("cnn_only_op", self._ctx(), 1.0)
        blended = ctrl.blend({"op_a": 0.5, "op_b": 0.5}, self._ctx())
        assert set(blended) == {"op_a", "op_b"}
        assert sum(blended.values()) == pytest.approx(1.0)

    def test_policy_roundtrip(self, tmp_path):
        ctrl = MetaController(alpha=0.7, warmup=5, tau=0.01)
        ctx = self._ctx()
        for i in range(7):
            ctrl.update("op_a", ctx, 0.02 * i)
        path = str(tmp_path / "policy.json")
        ctrl.save_policy(path)
        with open(path, encoding="utf-8") as f:
            assert json.load(f)["format"] == "dnaty.controller_policy"
        ctrl2 = MetaController.from_policy(path)
        assert ctrl2.n_obs == 7 and ctrl2.alpha == 0.7
        p1 = ctrl.select_probs(["op_a", "op_b"], ctx)
        p2 = ctrl2.select_probs(["op_a", "op_b"], ctx)
        assert p1["op_a"] == pytest.approx(p2["op_a"])

    def test_from_policy_rejects_wrong_dim(self):
        with pytest.raises(ValueError):
            MetaController.from_policy({"dim": 3, "operators": {}})


class TestEvolverControllerIntegration:
    def _loaders(self, n=192, input_size=12, n_classes=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, input_size, generator=g)
        y = torch.randint(0, n_classes, (n,), generator=g)
        ds = torch.utils.data.TensorDataset(x, y)
        return (torch.utils.data.DataLoader(ds, batch_size=64),
                torch.utils.data.DataLoader(ds, batch_size=64))

    def test_evolver_runs_with_controller_and_exports_policy(self):
        torch.manual_seed(0)
        np.random.seed(0)
        train, val = self._loaders()
        ev = DnatyEvolver(n_pop=4, n_generations=2, t_local=1,
                          input_size=12, n_classes=3, init_hidden=[16, 8],
                          verbose=False, controller=True)
        best, history = ev.run(train, val)
        assert len(history) == 2
        policy = ev.export_policy()
        assert policy["format"] == "dnaty.controller_policy"
        assert ev._controller.n_obs > 0, "controller received no reward feedback"

    def test_evolver_warm_policy(self):
        torch.manual_seed(0)
        np.random.seed(0)
        train, val = self._loaders()
        ev1 = DnatyEvolver(n_pop=4, n_generations=2, t_local=1,
                           input_size=12, n_classes=3, init_hidden=[16, 8],
                           verbose=False, controller=True)
        ev1.run(train, val)
        policy = ev1.export_policy()
        ev2 = DnatyEvolver(n_pop=4, n_generations=1, t_local=1,
                           input_size=12, n_classes=3, init_hidden=[16, 8],
                           verbose=False, controller=True,
                           controller_policy=policy)
        assert ev2._controller.n_obs == policy["n_obs"]
        assert ev2._controller.trust > 0.0

    def test_export_policy_without_controller_raises(self):
        ev = DnatyEvolver(n_pop=2, n_generations=1, verbose=False)
        with pytest.raises(ValueError):
            ev.export_policy()

    def test_controller_off_by_default_and_unchanged_behaviour(self):
        ev = DnatyEvolver(n_pop=2, n_generations=1, verbose=False)
        assert ev._controller is None
        probs = {"a": 0.5, "b": 0.5}
        assert ev._blend_probs(probs) is probs


# ---------------------------------------------------------------------------
# 4. Federated prior merging
# ---------------------------------------------------------------------------

class TestMergePriors:
    def _prior(self, scores, n_exp=10):
        return {"format": "dnaty.operator_prior", "version": 1, "gamma": 0.99,
                "n_experiences": n_exp, "scores": scores}

    def test_merge_combines_disjoint_strengths(self):
        pa = self._prior({"op_a": 10.0, "op_b": 1.0, "op_c": 1.0})
        pb = self._prior({"op_a": 1.0, "op_b": 8.0, "op_c": 1.0})
        merged = merge_priors([pa, pb])
        s = merged["scores"]
        assert merged["merged_from"] == 2
        assert s["op_a"] > s["op_c"] and s["op_b"] > s["op_c"]

    def test_normalisation_prevents_long_run_dominance(self):
        # Same relative preference, wildly different magnitudes -> equal say.
        pa = self._prior({"op_a": 1000.0, "op_b": 0.0})
        pb = self._prior({"op_a": 0.0, "op_b": 0.001})
        s = merge_priors([pa, pb])["scores"]
        assert s["op_a"] == pytest.approx(s["op_b"])

    def test_weights_bias_the_consensus(self):
        pa = self._prior({"op_a": 5.0, "op_b": 0.0})
        pb = self._prior({"op_a": 0.0, "op_b": 5.0})
        s = merge_priors([pa, pb], weights=[3.0, 1.0])["scores"]
        assert s["op_a"] > s["op_b"]

    def test_degenerate_priors_skipped(self):
        good = self._prior({"op_a": 3.0, "op_b": 1.0})
        flat = self._prior({"op_a": 2.0, "op_b": 2.0})
        empty = self._prior({})
        merged = merge_priors([good, flat, empty])
        assert merged["merged_from"] == 1

    def test_all_degenerate_raises(self):
        with pytest.raises(ValueError):
            merge_priors([self._prior({}), self._prior({"op_a": 1.0, "op_b": 1.0})])

    def test_mismatched_weights_raise(self):
        with pytest.raises(ValueError):
            merge_priors([self._prior({"op_a": 1.0, "op_b": 0.0})], weights=[1.0, 2.0])

    def test_merged_prior_seeds_memory(self):
        # op_a: best in pa (+1). op_b: worst in pa (-1) but best in pb (+1),
        # nets to 0 — mean-centring makes each prior vote on *relative*
        # preference. op_c: worst in pb (-0.5). Order: op_a > op_b > op_c.
        pa = self._prior({"op_a": 10.0, "op_b": 1.0})
        pb = self._prior({"op_b": 10.0, "op_c": 1.0})
        merged = merge_priors([pa, pb])
        mem = EpisodicMemory()
        n = mem.seed_from_prior(merged)
        assert n == 3
        probs = mem.query_mutation_probs(["op_a", "op_b", "op_c"])
        assert probs["op_a"] > probs["op_b"] > probs["op_c"]

    def test_bare_mapping_accepted(self):
        merged = merge_priors([{"op_a": 4.0, "op_b": 1.0}])
        assert merged["merged_from"] == 1


# ---------------------------------------------------------------------------
# End-to-end: compress(controller=True)
# ---------------------------------------------------------------------------

class TestCompressControllerE2E:
    def test_compress_with_controller_returns_policy(self):
        from dnaty import compress
        torch.manual_seed(0)
        np.random.seed(0)
        x = torch.randn(256, 20)
        y = torch.randint(0, 3, (256,))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=64
        )
        model = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )
        result = compress(model, loader, n_generations=2, n_pop=4,
                          finetune_epochs=0, verbose=False, seed=0,
                          controller=True)
        assert result.controller_policy.get("format") == "dnaty.controller_policy"
        # And the policy round-trips into a new controller
        ctrl = MetaController.from_policy(result.controller_policy)
        assert ctrl.n_obs == result.controller_policy["n_obs"]

    def test_compress_without_controller_has_empty_policy(self):
        from dnaty import compress
        torch.manual_seed(0)
        np.random.seed(0)
        x = torch.randn(192, 16)
        y = torch.randint(0, 2, (192,))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=64
        )
        model = nn.Sequential(nn.Linear(16, 48), nn.ReLU(), nn.Linear(48, 2))
        result = compress(model, loader, n_generations=1, n_pop=3,
                          finetune_epochs=0, verbose=False, seed=0)
        assert result.controller_policy == {}
