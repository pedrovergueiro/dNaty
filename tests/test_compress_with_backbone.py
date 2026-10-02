"""
Regression tests for compress_with_backbone():

1. The returned full model reproduces the reported accuracy. The head is trained
   on z-scored embeddings; the spliced head now carries that z-score. Before, the
   returned model fed raw features to it and predicted near chance (0.25 on four
   classes while reporting 0.85).
2. A head that was an nn.Sequential (Dropout + Linear, as in MobileNetV2 and
   EfficientNet) no longer crashes: the compressed DynamicMLP is spliced as a
   module, not unpacked into a Sequential that called its ModuleList.
3. Only the leading Dropouts of the original head are kept; activations between
   Linears (MobileNetV3) are not applied to features the head never saw.
4. val_data drives NAS selection and the reported accuracy, which is measured on
   the returned model in eval mode — also after end-to-end fine-tuning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty import compress_with_backbone
from dnaty._compress_helpers import StandardizedHead, _loader_accuracy

SEARCH = dict(n_generations=2, n_pop=3, verbose=False, seed=0, device="cpu")


class _Offset(nn.Module):
    """Embeddings far from zero mean / unit scale, as after a real CNN's ReLU."""

    def forward(self, x):
        return x * 50.0 + 10.0


def _data(n: int, seed: int):
    """Four classes: which quadrant of channel 0 is brightest."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, 16, 16, generator=g)
    quads = torch.stack(
        [
            x[:, 0, :8, :8].mean((1, 2)),
            x[:, 0, :8, 8:].mean((1, 2)),
            x[:, 0, 8:, :8].mean((1, 2)),
            x[:, 0, 8:, 8:].mean((1, 2)),
        ],
        1,
    )
    return x, quads.argmax(1)


def _features():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten(), _Offset()
    )


class ResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = _features()
        self.fc = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc(self.features(x))


class MobileNetV2Like(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = _features()
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 4))

    def forward(self, x):
        return self.classifier(self.features(x))


class MobileNetV3Like(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = _features()
        self.classifier = nn.Sequential(
            nn.Linear(32, 16), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@pytest.fixture(scope="module")
def loaders():
    torch.manual_seed(0)
    x_tr, y_tr = _data(1024, 1)
    x_va, y_va = _data(512, 2)
    train = DataLoader(TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(x_va, y_va), batch_size=128)
    return train, val


@pytest.mark.parametrize("backbone_cls", [ResNetLike, MobileNetV2Like, MobileNetV3Like])
def test_returned_model_reproduces_the_reported_accuracy(loaders, backbone_cls):
    train, val = loaders
    torch.manual_seed(0)
    result = compress_with_backbone(backbone_cls(), train, val_data=val, **SEARCH)

    measured = _loader_accuracy(result.model, val, "cpu")

    assert result.accuracy == pytest.approx(measured)
    assert measured > 0.5  # chance is 0.25; the unstandardized splice sat at ~0.25


def test_sequential_head_with_dropout_runs_and_keeps_the_dropout(loaders):
    train, _ = loaders
    torch.manual_seed(0)
    result = compress_with_backbone(MobileNetV2Like(), train, **SEARCH)
    head = result.model.classifier

    out = result.model.eval()(torch.rand(5, 3, 16, 16))

    assert out.shape == (5, 4)
    assert isinstance(head, StandardizedHead)
    assert [type(m) for m in head.pre] == [nn.Dropout]


def test_only_leading_dropouts_are_kept(loaders):
    """MobileNetV3's Hardswish sits between its Linears: the embeddings never went through it."""
    train, _ = loaders
    torch.manual_seed(0)
    result = compress_with_backbone(MobileNetV3Like(), train, **SEARCH)

    assert len(result.model.classifier.pre) == 0


def test_the_zscore_lives_in_the_state_dict(loaders):
    train, _ = loaders
    torch.manual_seed(0)
    result = compress_with_backbone(ResNetLike(), train, **SEARCH)
    state = result.model.state_dict()

    assert "fc.mean" in state and "fc.std" in state
    assert state["fc.mean"].abs().mean() > 1.0  # the offset features, not zero


def test_without_val_data_the_accuracy_is_the_returned_models_on_train(loaders):
    train, _ = loaders
    torch.manual_seed(0)
    result = compress_with_backbone(ResNetLike(), train, **SEARCH)

    assert result.accuracy == pytest.approx(_loader_accuracy(result.model, train, "cpu"))


def test_finetune_reports_the_eval_mode_accuracy_on_val(loaders):
    train, val = loaders
    torch.manual_seed(0)
    result = compress_with_backbone(
        MobileNetV2Like(), train, val_data=val, finetune_backbone=True, finetune_epochs=1, **SEARCH
    )

    assert result.model.training is False
    assert result.accuracy == pytest.approx(_loader_accuracy(result.model, val, "cpu"))
