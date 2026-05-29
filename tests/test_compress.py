import pytest
import torch
from dnaty.compress import Compressor

@pytest.fixture
def compressor():
    return Compressor(device="cpu")

def test_compress_basic(compressor):
    """Test basic compression on MNIST"""
    result = compressor.compress(
        model_name="resnet18",
        dataset="mnist",
        target_flops=0.5,
        epochs=5,
        population_size=10
    )
    assert result is not None
    assert "model" in result
    assert result["flops_reduction"] > 0

def test_compress_output_shape(compressor):
    """Test compressed model output shape matches original"""
    result = compressor.compress(
        model_name="mobilenet_v3_small",
        dataset="cifar10",
        target_flops=0.6,
        epochs=3,
        population_size=5
    )
    assert result["model"] is not None

def test_invalid_model():
    """Test error handling for invalid model"""
    compressor = Compressor(device="cpu")
    with pytest.raises(ValueError):
        compressor.compress(
            model_name="nonexistent_model",
            dataset="mnist",
            target_flops=0.5
        )

def test_compression_rate(compressor):
    """Test that compression rate matches target"""
    target_flops = 0.5
    result = compressor.compress(
        model_name="resnet50",
        dataset="imagenet",
        target_flops=target_flops,
        epochs=5,
        population_size=8
    )
    # Should be close to target (within 10%)
    assert 0.4 < result["flops_reduction"] < 0.6
