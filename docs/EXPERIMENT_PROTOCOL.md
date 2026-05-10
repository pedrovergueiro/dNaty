# dNaty v5.1 - Protocolo de experimentos

## Objetivo

Evitar mistura entre teste rápido, pré-validação e resultado científico.

## Níveis de execução

| Nível | Comando | Entra no paper? |
|-------|---------|-----------------|
| Smoke | `python -m experiments.run --profile smoke` | Não |
| Pré-validação | `python -m experiments.run --profile prevalidation` | Só como evidência interna |
| Full GPU | `python -m experiments.run --profile full_gpu` | Sim, se passar revisão |

## Critério mínimo para resultado científico

1. Ter `manifest.json`.
2. Ter `outputs/*.json`.
3. Ter 5+ seeds para benchmark principal.
4. Ter baseline justo.
5. Ter ablation.
6. Ter intervalo de confiança ou estatística pareada.
7. Ter commit registrado.

## Ablations obrigatórios

| Variante | Pergunta respondida |
|----------|---------------------|
| dNaty completo | Sistema final |
| Sem memória | A memória episódica importa? |
| Sem SAM | SAM melhora generalização? |
| Sem NSGA-II | Seleção multiobjetivo importa? |
| Mutação aleatória | Memória supera busca cega? |
| Arquitetura fixa | Evolução estrutural importa? |

## Estrutura de resultados

```text
results/
  runs/
    20260510T010000Z_prevalidation_exp2_cifar/
      config.json
      manifest.json
      outputs/
        exp2_cifar10_results.json
```

## Interpretação dos claims

- MNIST/FashionMNIST: evidência de eficiência e estabilidade.
- Split-MNIST: evidência de continual learning.
- CIFAR-10 v5.1: pré-validação até rodar full GPU.
- Dados reais tabulares: evidência de potencial comercial, ainda com baixo n.
