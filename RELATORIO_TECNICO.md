# Relatório Técnico — Tech Challenge Fase 2 (Projeto 1)

## 1. Objetivo

Melhorar a precisão/capacidade de detecção dos modelos de diagnóstico da Fase 1 por meio de **algoritmos genéticos** e tornar os resultados **interpretáveis** para profissionais de saúde com **LLMs**.

Contexto clínico: classificação binária de tumores (Benigno=0 / Maligno=1) a partir de quatro features com alta correlação com o diagnóstico.

## 2. Algoritmo genético

### 2.1 Representação genética

Cada indivíduo é um dicionário de hiperparâmetros (genes) no espaço de busca de KNN, SVM (`LinearSVC`) ou Random Forest.

### 2.2 Operadores

| Operador | Implementação |
|----------|---------------|
| Seleção | Torneio |
| Cruzamento | Uniforme por gene |
| Mutação | Substituição aleatória no domínio do gene |
| Elitismo | Preserva os melhores indivíduos |

Implementação em `algoritmo_genetico.py`, reutilizada pelo notebook e pelos testes.

### 2.3 Função fitness

**Recall** médio em validação cruzada estratificada (5 folds). A escolha prioriza redução de falsos negativos (casos malignos classificados como benignos).

### 2.4 Experimentos com configurações do AG

Foram definidos **três experimentos** com parâmetros distintos:

| Experimento | População | Gerações | Mutação | Cruzamento | Torneio | Elites |
|-------------|-----------|----------|---------|------------|---------|--------|
| Conservador | 8 | 10 | 0.05 | 0.70 | 2 | 1 |
| Equilibrado | 12 | 10 | 0.15 | 0.80 | 3 | 2 |
| Exploratório | 20 | 10 | 0.30 | 0.90 | 4 | 2 |

Os experimentos são executados no notebook `diagnosticos_algoritmo_genetico.ipynb` (modelo SVM como proxy de comparação das configs). A configuração vencedora alimenta a otimização final dos três classificadores.

**Resultados da execução (out/2026):** os três experimentos empataram em recall CV **0.9453** no SVM; por desempate de ordem, a config **Conservador** foi usada na otimização final. Hiperparâmetros SVM encontrados: `C=1`, `class_weight=balanced`, `max_iter=2000`.

| Experimento | População | Mutação | Recall CV (SVM) |
|-------------|-----------|---------|-----------------|
| Conservador | 8 | 0.05 | 0.9453 |
| Equilibrado | 12 | 0.15 | 0.9453 |
| Exploratório | 20 | 0.30 | 0.9453 |

### 2.5 Comparativo baseline × otimizado

O notebook registra métricas de teste (accuracy, precision, recall, F1) **antes** e **depois** da otimização genética, além de matrizes de confusão e curvas ROC.

**Recall no teste (métrica priorizada pelo fitness):**

| Modelo | Baseline | Otimizado AG | Ganho |
|--------|----------|--------------|-------|
| KNN | 0.8438 | 0.8594 | +0.0156 |
| SVM | 0.9375 | 0.9531 | +0.0156 |
| Random Forest | 0.8906 | 0.9219 | +0.0312 |

Accuracy no teste após AG: KNN 0.924 · SVM 0.959 · Random Forest 0.947. O pipeline exportado para a API é o SVM otimizado (`modelo_pipeline_genetico.pkl`).

## 3. API e monitoramento

- `main.py` carrega preferencialmente `modelo_pipeline_genetico.pkl`.
- Middleware registra método, rota, status HTTP e tempo de resposta (`X-Process-Time-Ms`).
- Endpoint `/health` expõe status e arquivo de modelo carregado.

## 4. Integração com LLMs

A chave da OpenAI é carregada via arquivo `.env` (`OPENAI_API_KEY`), documentado em `.env.example`. A chave real permanece fora do repositório Git.

Fluxo em `assistente_laudo_medico.ipynb`:

1. Texto do exame → prompt de **extração estruturada** (JSON dos 4 parâmetros).
2. Validação Pydantic → `GET /predict`.
3. Prompt de **laudo descritivo** (sem inventar diagnóstico/prognóstico).
4. Seção de **avaliação qualitativa** das respostas (clareza, fidelidade aos dados, ausência de alucinação, utilidade clínica educacional).

Prompts usam temperatura baixa, `response_format` JSON na extração e instruções explícitas de não inventar fatos clínicos.

## 5. Desafios e soluções

| Desafio | Solução |
|---------|---------|
| Custo computacional do AG | Cache de avaliações + experimentos curtos (10 gerações) para comparar configs |
| Falsos negativos | Fitness = recall |
| LLM inventando diagnóstico | Prompts restritivos + mapa de classes + disclaimer |
| API alinhada ao modelo otimizado | Servir `modelo_pipeline_genetico.pkl` com fallback documentado |

## 6. Como reproduzir

Ver [README.md](README.md). Validação automatizada: `pytest -q`.

## 7. Pendências externas

- Vídeo de demonstração (YouTube/Vimeo, até 15 minutos).
- (Opcional) Deploy em nuvem / IaC para pontuação extra.
