# Contributing to dNATY

Obrigado por considerar contribuir para o dNATY! Este documento fornece diretrizes e instruções.

## Como Começar

1. **Fork** o repositório
2. **Clone** seu fork: `git clone https://github.com/YOUR-USERNAME/dNATY.git`
3. **Create branch**: `git checkout -b feature/sua-feature`
4. **Make changes** e commit com mensagens claras
5. **Push** para seu fork e abra um **Pull Request**

## Código

### Estrutura
```
dNATY/
├── dnaty/              # Core package
│   ├── compress.py     # API pública
│   ├── evolver.py      # NAS search
│   ├── pruner.py       # Pruning
│   └── quantizer.py    # Quantização
├── frontend/           # React app
├── tests/              # Testes
└── docker/             # Dockerfile
```

### Padrões
- **Python**: Black (88 char), type hints, docstrings
- **TypeScript**: Prettier, ESLint, strict mode
- **Commits**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

### Testing
```bash
# Backend
pytest tests/ -v

# Frontend
npm run test
```

## Reporting Issues

Use templates no GitHub:
- **Bug Report**: descreva o erro, passos para reproduzir, resultado esperado
- **Feature Request**: explique a ideia, casos de uso, benefício

## Comunidade

- **Discord**: https://discord.gg/dnaty (para discussões)
- **GitHub Discussions**: para perguntas
- **Issues**: para bugs e features

## Código de Conduta

Seja respeitoso. Zero tolerância para harassment.

Obrigado por contribuir! 🚀
