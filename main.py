"""API FastAPI para servir o modelo de diagnóstico otimizado."""

from __future__ import annotations

import logging
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

MODEL_PATH = Path(__file__).resolve().parent / "modelo_pipeline_genetico.pkl"
FALLBACK_MODEL_PATH = Path(__file__).resolve().parent / "modelo_pipeline.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("diagnostico_api")

model_state: dict[str, Any] = {
    "pipeline": None,
    "model_file": None,
}


def carregar_modelo() -> tuple[Any, str]:
    """Carrega o pipeline otimizado; usa o baseline se o genético não existir."""
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as arquivo:
            return pickle.load(arquivo), MODEL_PATH.name
    if FALLBACK_MODEL_PATH.exists():
        logger.warning(
            "Modelo genético não encontrado; usando fallback %s",
            FALLBACK_MODEL_PATH.name,
        )
        with FALLBACK_MODEL_PATH.open("rb") as arquivo:
            return pickle.load(arquivo), FALLBACK_MODEL_PATH.name
    raise FileNotFoundError(
        "Nenhum modelo encontrado. Esperado modelo_pipeline_genetico.pkl "
        "ou modelo_pipeline.pkl."
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pipeline, model_file = carregar_modelo()
    model_state["pipeline"] = pipeline
    model_state["model_file"] = model_file
    logger.info("Modelo carregado: %s", model_file)
    yield
    model_state["pipeline"] = None
    model_state["model_file"] = None


app = FastAPI(
    title="API de Diagnóstico Médico — FIAP Fase 2",
    description=(
        "Serve o modelo de classificação de tumores otimizado por "
        "algoritmo genético, com logging de desempenho."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    inicio = time.perf_counter()
    resposta = await call_next(request)
    duracao_ms = (time.perf_counter() - inicio) * 1000
    logger.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        resposta.status_code,
        duracao_ms,
    )
    resposta.headers["X-Process-Time-Ms"] = f"{duracao_ms:.2f}"
    return resposta


@app.get("/")
def read_root():
    return {
        "message": "API de diagnóstico FIAP Fase 2",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    """Monitoramento básico de disponibilidade e modelo carregado."""
    pipeline = model_state.get("pipeline")
    ok = pipeline is not None
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "model_file": model_state.get("model_file"),
    }


@app.get("/predict")
def make_predict(
    concave_points_mean: float = Query(..., ge=0),
    concave_points_worst: float = Query(..., ge=0),
    perimeter_worst: float = Query(..., ge=0),
    radius_worst: float = Query(..., ge=0),
):
    pipeline = model_state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    novo_input = [
        [
            float(concave_points_mean),
            float(concave_points_worst),
            float(perimeter_worst),
            float(radius_worst),
        ]
    ]

    try:
        predicao = pipeline.predict(novo_input)
        classe = str(predicao[0])
    except Exception as erro:  # noqa: BLE001 — registra e devolve erro controlado
        logger.exception("Falha na predição")
        raise HTTPException(status_code=500, detail="Erro ao executar predição.") from erro

    logger.info(
        "Predicao concluida | classe=%s | entrada=%s",
        classe,
        {
            "concave_points_mean": concave_points_mean,
            "concave_points_worst": concave_points_worst,
            "perimeter_worst": perimeter_worst,
            "radius_worst": radius_worst,
        },
    )

    return {
        "output": classe,
        "model_file": model_state.get("model_file"),
        "mapa_classes": {"0": "Benigno", "1": "Maligno"},
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Erro não tratado em %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno inesperado."},
    )
