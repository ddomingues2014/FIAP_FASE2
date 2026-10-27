# Checklist — Tech Challenge Fase 2 (Projeto 1)

Status da entrega frente aos requisitos obrigatórios.

## 1. Otimização via Algoritmos Genéticos

- [x] Codificação genética dos hiperparâmetros (KNN, SVM, Random Forest)
- [x] Operadores: seleção por torneio, crossover uniforme, mutação, elitismo
- [x] Função fitness baseada em métrica (recall com validação cruzada estratificada)
- [x] Comparativo baseline × modelo otimizado (accuracy, precision, recall, F1)
- [x] Pelo menos 3 experimentos com configurações distintas do AG
- [x] Visualização da evolução do fitness por geração

## 2. Monitoramento, logging e arquitetura

- [x] Logging de requisições e tempo de resposta na API (`main.py`)
- [x] Endpoint de health/status para acompanhamento
- [x] Documentação de arquitetura no README e no relatório técnico
- [ ] (Opcional / pontuação extra) Deploy em nuvem + IaC

## 3. Integração com LLMs

- [x] Integração com LLM pré-treinada (OpenAI GPT)
- [x] Explicações em linguagem natural dos resultados do modelo
- [x] Prompt engineering adequado ao contexto médico/educacional
- [x] Avaliação qualitativa das interpretações geradas
- [x] Fluxo notebook → API de predição → laudo descritivo

## 4. Código e organização

- [x] Projeto Python com ambiente virtual (`venv` + `requirements.txt`)
- [x] Módulo reutilizável do algoritmo genético (`algoritmo_genetico.py`)
- [x] API FastAPI servindo o modelo otimizado
- [x] Testes automatizados (`tests/`)
- [x] README atualizado para a Fase 2
- [x] Relatório técnico (`RELATORIO_TECNICO.md`)
- [ ] Vídeo de demonstração (YouTube/Vimeo, até 15 min) — entrega externa

## Como validar localmente

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
pytest -q
uvicorn main:app --reload
```
