# dNATY — Benchmarks reais (medidos)

> Todos os números abaixo foram **medidos**, não estimados. Hardware: 6 núcleos de CPU.
> Config padrão: `n_generations=30, n_pop=15, t_local=3`. Acurácia em conjunto de teste
> **separado** (held-out), não no treino. Reproduzível com os scripts em `scripts/`.
> Data da medição: 2026-06.

## Resultados por dataset

| Dataset (real) | Linhas | Features | Classes | init_hidden | Tempo | FLOPs ↓ | Acurácia |
|---|---|---|---|---|---|---|---|
| MNIST | 10.000 | 784 | 10 | [512,256] | 4,1 min | **−50.4%** | 97.0% |
| Fashion-MNIST | 10.000 | 784 | 10 | [512,256] | 4,1 min | **−54.6%** | 86.4% |
| UCI Wine Quality (red) | 1.599 | 11 | 6 | [256,128] | 37 s | **−78.4%** | 63.7% |
| UCI Adult / Census Income | 10.000 | 104 | 2 | [256,128] | 3,6 min | −2.7% | 84.0% |
| UCI Covertype | 10.000 | 54 | 7 | [256,128] | 4,1 min | −1.5% | 78.1% |
| Social Friction (students vs workers) | 400 | 14 | 3 | [128,64] | 13 s | −6.9% | 100% |
| Social Friction v2 | 500 | 16 | 2 | [128,64] | 17 s | −5.4% | 85.0% |
| Indonesian Youth Digital Friction | 1.000 | 2 | 2 | [128,64] | 27 s | −1.5% | 99.5% |
| CIFAR-10 (MLP) | 10.000 | 3.072 | 10 | [512,256] | 13,4 min | −1.2% | 46.4% |

Fontes dos datasets:
- MNIST / Fashion-MNIST / CIFAR-10 — torchvision (datasets reais padrão)
- UCI Adult, Covertype, Wine Quality — UC Irvine ML Repository (download direto, sem auth)
- Social Friction / Indonesian Youth — datasets reais de comportamento em redes sociais (CSV)

## Como ler estes números (honesto)

**A magnitude da compressão depende de quanta capacidade redundante o modelo inicial tem.**
Não é "−50% em qualquer modelo":

- **Modelo superdimensionado para a tarefa → corte grande.**
  MNIST (−50%), Fashion-MNIST (−55%), Wine (−78%). O MLP inicial tinha gordura; o NAS achou
  uma rede muito menor que mantém a acurácia.

- **Modelo já bem dimensionado → corte pequeno.**
  Adult (−2.7%), Covertype (−1.5%). Isso é o comportamento **correto** de uma busca Pareto:
  ela não deve inflar nem quebrar um modelo que já é eficiente. "Pouco corte" aqui significa
  "seu modelo já estava perto do tamanho certo".

- **Limite real — só MLP hoje.**
  CIFAR-10 com MLP trava em ~46% (imagem RGB precisa de convolução). dNaty hoje é forte em
  dados MLP-friendly (tabular, imagem simples); busca em espaço de convoluções é trabalho futuro.

## Claim defensável (o que dá pra afirmar sem ser crucificado)

> "O dNaty encontra a menor arquitetura que mantém a acurácia. Em modelos superdimensionados
> isso é um corte de 50%+ de FLOPs (medido: MNIST −50%, Fashion-MNIST −55%, em ~4 min numa CPU
> de 6 núcleos). Em modelos já enxutos ele corta pouco — de propósito. Hoje é MLP-only;
> imagem RGB complexa via conv net é trabalho futuro."

## Reproduzir

```bash
python scripts/real_benchmarks.py       # MNIST/Fashion via compress() + tabulares sociais
python scripts/real_benchmarks_uci.py   # UCI Adult / Covertype / Wine
python scripts/fetch_real_datasets.py   # baixa os UCI (sem auth)
```
