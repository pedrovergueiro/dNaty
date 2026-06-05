# Roadmap v1.1.x

## v1.1.0 — API pública + persistência

### 1. Corrigir `target_flops` (bug — parâmetro aceito mas ignorado)
**Arquivo:** `dnaty/compress.py:111`

`lambda2 = 3e-6` é hardcoded; `target_flops` não influencia nada.
Mapeamento proposto:
```python
lambda2 = max(1e-7, 5e-6 * (1.0 - target_flops))
# target_flops=0.3 → lambda2=3.5e-6 (pressão alta)
# target_flops=0.8 → lambda2=1e-6  (pressão baixa)
```

### 2. `CompressResult.save()` + `dnaty.load()`
**Arquivo:** `dnaty/compress.py` — adicionar métodos ao dataclass

Após 450 avaliações o resultado não pode ser perdido por falta de persistência.
```python
result.save("model_compressed.pt")
result = dnaty.load("model_compressed.pt")
```

### 3. ONNX export (`result.export_onnx(path, input_shape)`)
**Arquivo:** `dnaty/compress.py` — método no `CompressResult`

Passo final para deploy em hardware CPU-only (drones, câmeras, robôs).
Depende só de PyTorch, sem nova dep.
```python
result.export_onnx("model.onnx", input_shape=(1, 784))
```

### 4. Expor `progress_callback` no `compress()` público
**Arquivo:** `dnaty/compress.py:132` — passar para `evolver.run()`

Já existe no `DnatyEvolver.run()` mas não é forwarded pelo wrapper público.

---

## v1.1.1 — Feedback da comunidade (r/computervision + r/deeplearning — 3 jun 2026)

### Fontes
- **Mechanical-Flatbed** (r/computervision)
- **Helix_roster13** (r/computervision)
- **Inner-Image-6313** (r/deeplearning)
- Contexto geral r/computervision

---

### 1. Residual connections como operador de mutação nativo
**Prioridade: alta** | Mencionado por: Mechanical-Flatbed + contexto geral

Hoje `add_skip` existe em `dnaty/operators/mutations.py:91` mas é uma skip connection genérica com projeção linear.
Implementar `add_residual` como operador dedicado: conexão identidade entre camadas de mesmo tamanho, sem projeção, como ResNet faz.
- Não confundir com `add_skip` atual (que suporta dimensões diferentes via projeção)
- Residual puro: `out = layer(x) + x` (exige `in_features == out_features`)

### 2. FLOPs counter por tipo de operação (não só por tamanho)
**Prioridade: alta** | Mencionado por: Mechanical-Flatbed + contexto geral

Hoje `count_flops()` em `dnaty/core/arch.py:77` conta apenas `2 * in * out` por camada Linear.
Conv2d tem custo completamente diferente: `2 * k*k * Cin * Cout * H * W`.
Implementar cost model correto por tipo:
- `nn.Linear`: `2 * in * out`
- `nn.Conv2d`: `2 * k*k * Cin * Cout * Hout * Wout`
- `DepthwiseSepConv`: `2 * k*k * C * H * W + 2 * C * Cout * H * W`
- Usar PyTorch hooks ou `torch.profiler` por camada, não só shape

### 3. DwConv / Depthwise Separable como operador de mutação
**Prioridade: alta** | Mencionado por: Helix_roster13

`depthwise_sep` em `dnaty/operators/mutations.py:223` é um placeholder para MLP — adiciona camada estreita mas não é DwConv real.
Implementar operador CNN real:
- Substituir `Conv2d(Cin, Cout, k)` por `DwConv(Cin, k) → PwConv(Cin, Cout)`
- FLOPs ~8–9x menores que conv padrão com mesmo Cin/Cout/k
- Já existe base em `dnaty/operators/mutations_cnn.py` para expandir

### 4. Suporte inicial a arquiteturas conv (além de MLP)
**Prioridade: média** | Mencionado por: contexto geral + Helix_roster13

`CnnEvolver` existe em `dnaty/evolution/evolver.py:260` mas `DynamicCNN` é WIP.
Próximos passos concretos:
- Estabilizar `DynamicCNN` para CIFAR-10 (classificação, não detecção ainda)
- Expor `compress_cnn()` na API pública paralelo ao `compress()` atual
- **Não** afirmar equivalência com OFA/MnasNet nessa fase (ver `dnaty_honest_scope.md`)

### 5. Benchmark em Raspberry Pi (Raspberry Pi 4, CPU-only)
**Prioridade: média** | Mencionado por: Helix_roster13 (meta: 25+ FPS)

Benchmark real de latência — não só FLOPs teóricos.
- Testar modelo comprimido vs original em RPi 4 (ARM Cortex-A72, 4GB)
- Medir latência por inferência (ms) e FPS real para input 28×28 (MNIST) e 32×32 (CIFAR)
- Documentar em `BENCHMARKS_EDGE.md` no mesmo formato de `BENCHMARKS_REAL.md`
- VisDrone seria passo seguinte (detecção de objetos, escopo maior)

### 6. Data drift detection básico
**Prioridade: baixa (v1.2.x)** | Mencionado por: Inner-Image-6313

Pipeline de produção além da compressão isolada.
- Detectar quando distribuição de input diverge do treino (PSI / KL divergence simples)
- Trigger de recompressão automática quando drift > threshold
- Tracking de falhas e edge cases em deployment
- Depende de instrumentação no lado do usuário — escopo maior, não é só biblioteca

---

## Notas de implementação

- `target_flops` fix é pré-requisito para v1.1.0 — sem ele o parâmetro principal da API é mentira
- ONNX export não adiciona dependências novas
- FLOPs counter correto (item 2 do v1.1.1) é pré-requisito para DwConv e CNN NAS fazerem sentido
- Residual connections e DwConv são independentes entre si — podem ser implementados em paralelo
- Data drift detection é escopo de v1.2.x, não cabe em v1.1.1
