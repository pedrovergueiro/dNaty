# dNaty v5.1 - Manual leigo para executar e salvar resultados

Este manual mostra o caminho certo para usar o dNaty sem baguncar resultados.

## 1. Instalar o ambiente

Abra o PowerShell na pasta do projeto:

```powershell
cd C:\Users\pedro\OneDrive\Documentos\projetos_pessoal\dNATY
```

Crie e ative um ambiente virtual:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se estiver usando seu Python atual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale o projeto como biblioteca local:

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 2. Baixar/preparar datasets

Rode:

```powershell
python -m experiments.prepare_datasets
```

Isso deixa MNIST, FashionMNIST e CIFAR-10 em `data/`.
Se já estiverem baixados, o comando só confere/cacheia.

## 3. Rodar teste rápido antes de qualquer experimento

```powershell
python -m experiments.run --profile smoke --notes "checagem inicial"
```

O resultado correto é terminar com:

```text
[OK] Run saved at: results/runs/...
```

Cada execução cria uma pasta em `results/runs/` com:

- `config.json`: configuração usada.
- `manifest.json`: versão, commit, ambiente, comando e status.
- `outputs/`: cópias dos JSONs gerados, quando houver.

## 4. Rodar CIFAR pré-validação

Use isto para testar o pipeline sem prometer resultado científico final:

```powershell
python -m experiments.run --profile prevalidation --experiment exp2_cifar --notes "CIFAR v5.1 prevalidacao"
```

Importante: esta rodada ainda é pré-validação. Para artigo externo, precisa de GPU, 5+ seeds e ablation.

## 5. Rodar experimento completo para artigo

Use quando tiver GPU/tempo:

```powershell
python -m experiments.run --profile full_gpu --notes "rodada completa para paper"
```

Esse perfil mira:

- MNIST/FashionMNIST
- CIFAR-10
- Split-MNIST Continual Learning
- Mais seeds
- Configuração mais próxima de publicação

## 6. Como usar resultado no artigo

Nunca copie número direto do terminal. Use sempre:

```text
results/runs/<data>_<experimento>/manifest.json
results/runs/<data>_<experimento>/outputs/*.json
```

O `manifest.json` prova:

- qual comando foi rodado
- qual config foi usada
- qual commit gerou o resultado
- se o repositório estava sujo
- qual versão do dNaty foi usada
- qual máquina/ambiente executou

## 7. Regras para não se enganar

- `smoke`: só valida que o sistema liga.
- `prevalidation`: gera evidência interna, ainda não é paper final.
- `full_gpu`: candidato a resultado científico.
- Resultado sem `manifest.json` não entra no artigo.
- Resultado hardcoded/projetado deve ser marcado como pré-validação.

## 8. Como usar como biblioteca

Depois de `pip install -e .`, outro dev pode importar:

```python
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.experiments.fast_dataset import FastDataset

dataset = FastDataset("MNIST", device="cpu")
evolver = DnatyEvolver(n_pop=8, n_generations=10, t_local=3)
best, history = evolver.run(dataset, dataset)
```

## 9. Próximo passo de produto

Para virar SaaS real:

1. Criar FastAPI.
2. Persistir runs no PostgreSQL.
3. Criar Model Registry.
4. Criar fila de treino.
5. Criar Drift Monitor.
6. Promover modelos só se passarem em métricas e rollback.
