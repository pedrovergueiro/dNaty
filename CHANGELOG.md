# Changelog

Todas as mudanças notáveis neste projeto estão documentadas neste arquivo.

## [5.2.0] - 2026-05-29

### Added
- **Public API**: `compress()` function para uso direto
- **Website**: React frontend com admin/user dashboard
- **Docker**: Multi-stage Dockerfile com CPU/GPU variants
- **Admin Panel**: Real-time metrics (compressions, uptime, FLOPs reduction)
- **User Dashboard**: Plano info, quota tracking, compression history
- **Pricing Panel**: 3 tiers (Starter $29, Pro $99, Enterprise custom)
- **Benchmarks**: CIFAR-100 baseline (-46.5% FLOPs vs EWC)
- **Documentation**: API docs + architecture guide

### Changed
- MobileNetV3 classifier reconstruction para CIFAR-100 compatibility
- Frontend navigation com React Router (fixes: dashboard, login, admin, pricing links)

### Fixed
- Dimension mismatch error em MobileNetV3 classifier layer
- React app not loading (added #root mounting point + script)
- Missing API imports e service stubs

## [5.1.0] - 2026-05-24

### Added
- Evolutionary NAS search com 6.9x speedup vs EWC
- Logit distillation in contrastive learning
- AMP (Automatic Mixed Precision) support
- Data augmentation (RandomCrop, RandomFlip)
- Full CIFAR-100 benchmark suite

### Benchmark Results
- **ResNet-50**: -46.5% FLOPs, 1.6x speedup
- **EfficientNet-B0**: -40% FLOPs
- **MobileNetV3-Large**: -98% FLOPs (edge device)

## [5.0.0] - 2026-05-01

### Initial Release
- Core compress() API
- Evolutionary search algorithm
- Pruning + quantization pipeline
- MNIST/CIFAR-10 support
- CLI interface

---

### Roadmap Futuro
- **v5.3**: Mixed precision distillation
- **v5.4**: Multi-GPU training
- **v6.0**: Custom architecture search
