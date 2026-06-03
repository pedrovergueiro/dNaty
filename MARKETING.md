# dNATY — Posicionamento, Mensagem e Conteúdo

> Documento de trabalho. Copy pronta para usar. Foco: clareza acima de hype.
> Honestidade: o site atual **já está bom** no ângulo Edge ML. O buraco real era
> nunca responder "como difere de pruning / quantização / distillation / NAS".
> Isso foi corrigido na landing (seção 04) e no README.

---

## 1. Declaração de posicionamento

**Para** times de engenharia que precisam rodar modelos de visão em dispositivos sem GPU
(câmeras, drones, robôs, sensores industriais),
**o dNATY** é uma ferramenta de compressão de modelos
**que** encontra automaticamente uma arquitetura menor que faz o mesmo trabalho —
rodando só em CPU, em minutos, sem retreinar.
**Diferente de** pruning, quantização ou distillation (que encolhem o modelo que você já tem),
o dNATY **redesenha a arquitetura** — e você ainda pode quantizar por cima.
**Importa porque** modelo menor = inferência mais barata, menos lag, e cabe no
hardware que já está no dispositivo — sem cloud, sem NVIDIA.

### Frases de 1 linha (teste dos 10 segundos)

- **"Rode IA em qualquer dispositivo. Sem GPU."** ← (hero atual, manter)
- "Modelos menores, inferência mais barata — uma chamada de função."
- "Não é pruning nem quantização. O dNATY redesenha o modelo pra caber na borda."
- "Seu modelo de visão rodando em tempo real numa câmera — sem cloud, sem NVIDIA."

---

## 2. Análise seção por seção

### Hero
- **Problema atual:** nenhum — "Ship AI on any device. No GPU required." já é outcome-first e claro.
- **Por que confunde:** não confunde. Mantém.
- **Recomendação:** manter. Único ajuste opcional: adicionar prova social/logos quando houver.

### Descrição de produto / Features
- **Problema atual:** as 4 features já são benefícios ("Runs on the device", "No cloud, no latency", "No retraining", "One-line API"). Bom.
- **Por que confunde:** o termo "compress" faz o engenheiro de ML pensar imediatamente em pruning/quantização — e a landing nunca dizia que é diferente.
- **Versão melhorada:** ADICIONADA seção "04 — How it's different" que nomeia pruning/quantização/distillation/DARTS/RandomNAS e mostra que são complementares.
- **Copy final (já no site):**
  > **Not pruning. Not quantization. It redesigns the model.**
  > Most methods shrink the model you already have. dNATY searches for a smaller
  > architecture that does the same job — and you can still quantize it afterward.
  > **They stack.** Run dNATY, then quantize to int8 — the savings multiply.

### README — introdução
- **Problema atual (antes):** abria com o método ("episodic memory-guided evolutionary search") antes do benefício.
- **Por que confunde:** o leitor técnico quer saber *o que ganha* antes de *como funciona*.
- **Versão melhorada (já aplicada):** abre com "The problem" + "What you get", depois a tabela de diferença vs métodos, e só então o "como funciona".

### Onboarding do dashboard (recomendação — copy pronta)
O dashboard hoje cai direto na lista de treinos. Sugestão de primeiro acesso (empty state):

- **Título:** "Comprima seu primeiro modelo"
- **Subtítulo:** "Suba um dataset (CSV, imagens em ZIP, qualquer formato) ou use o MNIST. Em minutos você tem um modelo menor pra rodar na borda."
- **3 passos visuais:**
  1. Escolha ou suba um dataset
  2. dNATY procura a menor arquitetura que mantém a acurácia (roda em CPU)
  3. Baixe `.pth`/`.onnx` e mande pro dispositivo
- **CTA primário:** "Treinar agora →"  ·  **secundário:** "Ver exemplo (MNIST)"

### CTA final
- **Problema atual:** "Get your model running on the device." — já é bom.
- **Recomendação:** manter. Opcional: trocar "Try compress()" por "Ver um resultado real →" linkando /benchmarks (reduz fricção pra quem ainda não confia).

---

## 3. Posicionamento vs cada alternativa (resumo de vendas)

| Pergunta do cliente | Resposta de 1 frase |
|---|---|
| "É tipo NAS?" | É NAS evolutivo — mas roda em CPU, sem GPU e sem horas de config (diferente de DARTS). |
| "É tipo quantização?" | Não. Quantização troca a precisão dos pesos; o dNATY troca a arquitetura. **Use os dois juntos.** |
| "É tipo pruning?" | Pruning zera pesos (precisa de runtime esparso pra acelerar). O dNATY entrega uma rede densa menor que já roda rápido em CPU. |
| "É tipo distillation?" | Distillation exige você desenhar o aluno e treinar. O dNATY acha a arquitetura sozinho, sem loop de treino manual. |
| "Por que não DARTS?" | DARTS precisa de GPU e configuração. dNATY: uma chamada, CPU. |
| "Por que não Random NAS?" | Random NAS não lembra o que funcionou; o dNATY usa memória episódica e converge mais rápido. |

---

## 4. Redes sociais — modelos de comentário (chamar pro perfil)

> Use em posts de ML/IA de outras pessoas. Soa como contribuição, não spam.
> Troque `@seu_perfil` pelo seu handle real. Site: **dnaty.org**

**LinkedIn (técnico, gera autoridade):**
- "Ótimo ponto sobre custo de inferência. A gente atacou isso por outro ângulo: em vez de quantizar, deixar a busca evolutiva achar uma arquitetura menor que mantém a acurácia — roda só em CPU. Comento mais sobre Edge ML no meu perfil 👉 @seu_perfil"
- "Esse é exatamente o problema de rodar modelo em câmera/drone sem GPU. Tenho compartilhado experimentos reais (benchmarks reproduzíveis) no @seu_perfil pra quem trabalha com Edge ML."

**Instagram / Threads (mais leve):**
- "Modelo de visão rodando em tempo real sem GPU é possível 👀 Posto código e resultados de ML no @seu_perfil — dá uma olhada se curte a área."
- "Spoiler: dá pra cortar ~46% do custo de inferência sem perder acurácia. Explico no @seu_perfil 🚀"

**X / Twitter:**
- "quantização não é a única forma de encolher um modelo. dá pra procurar uma arquitetura menor direto. compartilho experimentos de Edge ML no @seu_perfil"

**Reddit (r/MachineLearning, r/learnmachinelearning) — sem soar promo:**
- "If your bottleneck is inference cost on CPU/edge, architecture search (not just quantization/pruning) is underrated. Reproducible benchmarks (one script) at dnaty.org if you want to poke holes."

> ⚠️ Honestidade: em comunidades técnicas (Reddit/HN) auto-promoção explícita é mal vista.
> Contribua com valor real primeiro; o link pro perfil vem como consequência.

---

## 5. Discord — estrutura e mensagens

**Convite atual:** https://discord.gg/WQ3dNwUu

### Estrutura de canais sugerida
```
📢 BOAS-VINDAS
  #regras
  #anúncios
💬 COMUNIDADE
  #apresente-se
  #geral
  #vagas-ml
🧠 MACHINE LEARNING
  #news-ml          ← notícias e papers
  #códigos-e-snippets
  #dicas-e-tutoriais
  #dúvidas
🛠️ dNATY
  #suporte
  #feedback
  #showcase         ← resultados/projetos da galera
```

### Mensagem de boas-vindas (canal #regras ou bot)
> 👋 Bem-vindo(a)! Aqui a gente fala de **Machine Learning na prática** — código, papers, e como rodar IA sem precisar de GPU cara.
> • Apresente-se em #apresente-se
> • Dúvida de ML? #dúvidas
> • Usando o dNATY? #suporte e #showcase
> Sem spam, sem flood. Respeito sempre. Bom proveito 🚀

### Template de anúncio (#anúncios)
> 🚀 **[Novidade]** {título}
> {1-2 frases do que mudou e por que importa pra você}
> 🔗 {link}
> Dúvidas? Manda em #suporte.

---

## 6. Canal do WhatsApp — ML (news + códigos + dicas)

### Nome (opções)
- **"ML na Real"** — sem hype, foco prático
- "Edge ML BR"
- "Machine Learning Diário"

### Descrição do canal
> 🤖 Machine Learning na prática, em português.
> Notícias que importam · snippets de código que você usa hoje · dicas pra rodar IA sem GPU cara.
> Sem enrolação. Por @seu_perfil / Vergueiro Tech · dnaty.org

### Cadência sugerida
- **Seg:** notícia/paper da semana (1 parágrafo + por que importa)
- **Qua:** snippet de código (PyTorch / sklearn / Edge ML)
- **Sex:** dica prática (1 truque que economiza tempo ou custo)

### Modelos de post

**📰 Notícia (segunda):**
> 📰 *ML desta semana*
> {manchete em 1 linha}
> Por que importa: {1-2 frases — impacto prático pra quem constrói}
> 🔗 {link}

**💻 Código (quarta):**
> 💻 *Snippet da semana — {tema}*
> ```python
> # Ex: comprimir um modelo pra rodar na borda
> from dnaty import compress
> result = compress(model, dataset, target_flops=0.5)
> print(result.flops_reduction_pct)  # ~46%
> ```
> Quando usar: {1 frase}
> Salva esse 👆

**💡 Dica (sexta):**
> 💡 *Dica de ML*
> {dica em 1-2 frases}
> Exemplo: "Antes de comprar GPU pra inferência, meça quanto custa em CPU.
> Muita vez um modelo menor resolve — e roda em qualquer lugar."

### Exemplos prontos de conteúdo (primeiras 3 semanas)

1. **News:** "Modelos pequenos estão ganhando: a indústria percebeu que rodar IA na borda corta custo e latência. Por que importa: nem todo problema precisa de um LLM gigante na nuvem."
2. **Código:** snippet de `compress()` acima.
3. **Dica:** "Quantização (int8) + arquitetura menor não competem — somam. Encolha a arquitetura primeiro, depois quantize."
4. **News:** resumo de um paper de NAS/edge da semana.
5. **Código:** `torch.quantization` em 5 linhas.
6. **Dica:** "Meça FLOPs, não só parâmetros — é o que define o custo real de inferência."

---

## 7. Checklist de execução

- [x] Landing: seção "How it's different" (vs pruning/quant/distillation/NAS) — **feito**
- [x] README: intro outcome-first + tabela de diferença — **feito**
- [ ] Dashboard: empty-state de onboarding (copy na seção 2)
- [ ] Adicionar link do perfil pessoal no footer/Discord
- [ ] Criar canal WhatsApp com nome + descrição acima
- [ ] Agendar primeira semana de posts (modelos na seção 6)
