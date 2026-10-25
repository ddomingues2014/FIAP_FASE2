# Tech Challenge FIAP — Fase 2 (Projeto 1)

Otimização de modelos de diagnóstico médico com **algoritmos genéticos** e interpretação dos resultados com **LLMs** (OpenAI), evoluindo o modelo da Fase 1.

## Autores

- Denilson Dominguez — domingues.denilson2014@gmail.com
- Lucas da Silva Torres Martins — lucas.hayim@gmail.com
- Lucas Nascimento Lopes — lucaslsantos159@gmail.com
- Uriel Alves da Silva — lioneluri@hotmail.com
- Vinicius Breda Silva — vinicius_breda_silva@hotmail.com

## O que este projeto entrega

1. **Algoritmo genético** para otimizar hiperparâmetros de KNN, SVM e Random Forest (`algoritmo_genetico.py` + `diagnosticos_algoritmo_genetico.ipynb`).
2. **Pelo menos 3 experimentos** com configurações distintas do AG (Conservador, Equilibrado, Exploratório).
3. **Comparativo** entre modelos originais (baseline) e otimizados.
4. **API FastAPI** servindo `modelo_pipeline_genetico.pkl`, com logging e `/health`.
5. **Assistente LLM** que gera laudo descritivo a partir do resultado do modelo (`assistente_laudo_medico.ipynb`).
6. **Testes automatizados** em `tests/`.

## Arquitetura

```text
[dados/dados.csv]
        |
        v
[diagnosticos_algoritmo_genetico.ipynb]
  - EDA + baseline
  - 3 experimentos AG
  - treino final otimizado
        |
        v
[modelo_pipeline_genetico.pkl]
        |
        v
[main.py  FastAPI + logging]
        ^
        |
[assistente_laudo_medico.ipynb]
  texto exame -> OpenAI (extração)
              -> /predict
              -> OpenAI (laudo)
```

## Estrutura

```text
FIAP_FASE2/
├── algoritmo_genetico.py              # Operadores e experimentos do AG
├── assistente_laudo_medico.ipynb      # Integração LLM + laudo
├── diagnosticos.ipynb                 # Baseline Fase 1
├── diagnosticos_algoritmo_genetico.ipynb
├── main.py                            # API de predição
├── modelo_pipeline.pkl                # Baseline
├── modelo_pipeline_genetico.pkl       # Modelo otimizado (servido pela API)
├── dados/dados.csv
├── tests/
├── .env.example                       # Modelo de config da OpenAI (sem chave real)
├── .env                               # Chave real (local; não versionado)
├── CHECKLIST.md
├── RELATORIO_TECNICO.md
└── requirements.txt
```

## Como executar

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# .\venv\Scripts\Activate.ps1   # Windows

pip install -r requirements.txt
pytest -q
uvicorn main:app --reload
```

API: http://127.0.0.1:8000  
Docs: http://127.0.0.1:8000/docs  

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/` | Informações da API |
| GET | `/health` | Status e modelo carregado |
| GET | `/predict` | Predição (4 features clínicas) |

Parâmetros de `/predict`: `concave_points_mean`, `concave_points_worst`, `perimeter_worst`, `radius_worst`.

### Assistente LLM (OpenAI)

A integração com LLM já está no projeto (`assistente_laudo_medico.ipynb`). A chave **não** vai no Git — fica só no `.env` local (exigência usual dos professores: integração + configuração segura).

1. Suba a API: `uvicorn main:app --reload`
2. Configure a chave:
   ```bash
   cp .env.example .env
   # edite .env e cole OPENAI_API_KEY=sk-...
   ```
3. Execute `assistente_laudo_medico.ipynb`

Arquivos:
- `.env` — chave real (local, no `.gitignore`)
- `.env.example` — modelo versionado para a banca/professores

## Documentação da entrega

- [CHECKLIST.md](CHECKLIST.md) — cobertura dos requisitos
- [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md) — decisões, experimentos e resultados

> Uso educacional / apoio à decisão: não substitui laudo médico.
