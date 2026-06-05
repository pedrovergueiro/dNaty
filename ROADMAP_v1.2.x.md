# Roadmap v1.2.x

Itens identificados pelo feedback da comunidade (r/computervision, r/deeplearning — jun 2026)
que exigem hardware externo, datasets específicos ou escopo de arquitetura maior.
Não cabem em v1.1.x.

---

## v1.2.0 — Object Detection + Edge Benchmarks

### 1. Suporte a modelos de detecção de objetos
**Solicitado por:** Helix_roster13 (r/computervision)

`compress()` e `compress_cnn()` hoje só suportam classificação.
Detecção requer:
- Representação de cabeças de detecção (anchor boxes, YOLOv8-like)
- Loss multipart: `cls_loss + box_loss + obj_loss`
- `DynamicDetector` — novo tipo além de `DynamicMLP` e `DynamicCNN`
- Mutações específicas: troca de backbone, neck, head separadamente

**Escopo estimado:** 3–4 semanas. Não afirmar equivalência com YOLOv8 nessa fase.

---

### 2. Benchmark Raspberry Pi 4 (CPU-only, real-time)
**Solicitado por:** Helix_roster13 (r/computervision) — meta: 25+ FPS

Latência teórica via `benchmark_latency()` já disponível em v1.1.0.
Benchmark real requer:
- Hardware: Raspberry Pi 4 (ARM Cortex-A72, 4 GB)
- Input 28×28 (MNIST) e 32×32 (CIFAR) → validar 25+ FPS
- ONNX runtime benchmark (além de PyTorch)
- Documentar em `BENCHMARKS_EDGE.md` no mesmo formato de `BENCHMARKS_REAL.md`

**Bloqueador:** acesso físico ao hardware.

---

### 3. VisDrone dataset — NAS para detecção em drones
**Solicitado por:** Helix_roster13 (r/computervision)

Pipeline completo:
- Download + preprocessing do VisDrone2019-DET
- `FastDataset("VisDrone")` com image→annotation loading
- `CnnEvolver` adaptado para detection loss
- Métrica: mAP@0.5 ao invés de accuracy

**Bloqueador:** depende do item 1 (detecção) + hardware para treino razoável.

---

### 4. Layer swapping guiado por FLOPs target
**Solicitado por:** contexto geral r/computervision

`swap_conv_to_dw` (já em v1.1.1) substitui aleatoriamente. Evoluir para:
- Selecionar a camada com maior FLOPs/parâmetro ratio para swap
- Budget-aware: parar quando `current_flops <= target_flops * original_flops`
- Integrar com `DnatyEvolver` como operador prioritário quando budget excedido

---

## v1.2.1 — Pipeline de Produção Avançado

### 5. Trigger de recompressão automática
**Solicitado por:** Inner-Image-6313 (r/deeplearning)

`DriftDetector` já detecta drift (v1.1.1). Próximo passo:
- `ProductionTracker.auto_retrigger(compress_fn, train_data)` — chama
  `compress()` automaticamente quando `psi_mean > threshold` por N batches consecutivos
- Integração com notificações (webhook, email) quando trigger disparar

---

### 6. Tracking de falhas e edge cases em deployment
**Solicitado por:** Inner-Image-6313 (r/deeplearning)

`ProductionTracker.record_outcome()` já existe (v1.1.1).
Expandir para:
- Clustering de inputs que causaram erros (PCA/t-SNE dos embeddings)
- `tracker.export_failure_report(path)` → JSON com amostras problemáticas
- Integração com bases de dados externas (SQLite local, PostgreSQL via URI)

---

## Notas de escopo

- **v1.1.x** foca em MLP compression + monitoring básico (classificação)
- **v1.2.x** foca em CNN + detecção + edge hardware real
- **Não afirmar** equivalência com OFA, MnasNet, YOLOv8 até benchmark real disponível
- VisDrone e RPi exigem sessão de trabalho com hardware físico
