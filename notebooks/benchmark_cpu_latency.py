"""
dNATY CPU Latency Benchmark
Mede inference time real: ResNet-50, EfficientNet-B0, MobileNetV3
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import json
import time
from pathlib import Path

# ===================================================================
# CONFIG
# ===================================================================
DEVICE = "cpu"
N_CLASSES = 1000  # ImageNet
BATCH_SIZES = [1, 4, 8, 16]
N_WARMUP = 5
N_ITER = 15
INPUT_SIZE = 224  # ImageNet standard


# Model Definitions
class ResNet50(nn.Module):
    def __init__(self, num_classes=10000):
        super().__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=False)
        if num_classes != 1000:
            self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = models.mobilenet_v3_large(pretrained=False)
        if num_classes != 1000:
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_latency(model, batch_size, n_warmup=5, n_iter=20):
    """
    Mede latencia de inference:
    - Warmup: descarta primeira execucao (cache)
    - Mede: N_ITER iteracoes
    Retorna: (latency_ms, throughput_imgs_per_sec, std_ms)
    """
    model.eval()

    # Input dummy
    x = torch.randn(batch_size, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    # Sincroniza CPU
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # em ms

    times = np.array(times)

    latency_ms = times.mean()
    latency_std = times.std()
    throughput = (batch_size * 1000) / latency_ms  # imgs/sec

    return {
        'latency_ms': round(latency_ms, 3),
        'latency_std': round(latency_std, 3),
        'latency_min': round(times.min(), 3),
        'latency_max': round(times.max(), 3),
        'latency_p95': round(np.percentile(times, 95), 3),
        'throughput_imgs_per_sec': round(throughput, 1),
        'times_raw': times.tolist()
    }


def main():
    print(f"\n{'='*70}")
    print("CPU LATENCY BENCHMARK - dNATY vs Baselines")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Classes: {N_CLASSES}")
    print(f"Warmup: {N_WARMUP}, Iterations: {N_ITER}\n")

    models_dict = {
        'ResNet-50': ResNet50(N_CLASSES),
        'EfficientNet-B0': EfficientNetB0(N_CLASSES),
        'MobileNetV3-Large': MobileNetV3Large(N_CLASSES),
    }

    results = {}

    for model_name, model in models_dict.items():
        print(f"\n{model_name}")
        print(f"{'-'*70}")

        model = model.to(DEVICE)
        params = count_params(model)
        print(f"Parameters: {params:,}\n")

        model_results = {
            'params': params,
            'batch_sizes': {}
        }

        for batch_size in BATCH_SIZES:
            latency = benchmark_latency(model, batch_size, N_WARMUP, N_ITER)
            model_results['batch_sizes'][batch_size] = latency

            print(f"  Batch {batch_size:2d}  | "
                  f"Latency: {latency['latency_ms']:7.2f}ms "
                  f"(+/-{latency['latency_std']:5.2f}) | "
                  f"Throughput: {latency['throughput_imgs_per_sec']:6.0f} imgs/sec")

        results[model_name] = model_results

    # Summary
    print(f"\n{'='*70}")
    print("RESUMO - Latencia por Imagem (Batch=1)")
    print(f"{'='*70}")

    for model_name in models_dict.keys():
        latency_b1 = results[model_name]['batch_sizes'][1]['latency_ms']
        params = results[model_name]['params']
        print(f"{model_name:20} | {latency_b1:6.2f}ms/img | {params:,} params")

    print(f"\n{'='*70}")
    print("RESUMO - Throughput (maximo)")
    print(f"{'='*70}")

    for model_name in models_dict.keys():
        max_throughput = max(
            results[model_name]['batch_sizes'][bs]['throughput_imgs_per_sec']
            for bs in BATCH_SIZES
        )
        print(f"{model_name:20} | {max_throughput:6.0f} imgs/sec")

    # Save results
    output = {
        'device': DEVICE,
        'input_size': INPUT_SIZE,
        'n_classes': N_CLASSES,
        'warmup_iterations': N_WARMUP,
        'benchmark_iterations': N_ITER,
        'batch_sizes': BATCH_SIZES,
        'results': results,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    results_dir = Path("results_cpu_latency")
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "cpu_latency_benchmark.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Resultados salvos em: {output_path}\n")


if __name__ == "__main__":
    main()
