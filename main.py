"""
API de predição para modelo de diagnóstico.

Decisões de implementação:
1. O modelo é carregado uma única vez durante a inicialização da aplicação.
2. As predições são recebidas por POST em formato JSON.
3. Logs registram request_id, rota, status e tempo de processamento.
4. Métricas são expostas no endpoint /metrics para coleta pelo Prometheus.
5. Endpoints /health/live e /health/ready suportam health checks.
"""

import logging
import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


# ============================================================
# Configuração
# ============================================================

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "modelo_pipeline.pkl")
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("diagnostico-api")


# ============================================================
# Métricas Prometheus
# ============================================================

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Quantidade total de requisições recebidas",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "Tempo de processamento das requisições",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Quantidade total de predições realizadas",
    ["prediction"],
)

PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Quantidade de erros durante a predição",
    ["error_type"],
)

MODEL_READY = Gauge(
    "model_ready",
    "Indica se o modelo foi carregado corretamente",
)


# ============================================================
# Estado da aplicação
# ============================================================

model = None


# ============================================================
# Inicialização e encerramento
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executa tarefas de inicialização e encerramento da API.

    O modelo é carregado uma única vez quando a aplicação sobe,
    evitando leitura do arquivo em cada requisição.
    """

    global model

    try:
        logger.info(
            "Carregando modelo | caminho=%s",
            MODEL_PATH,
        )

        with MODEL_PATH.open("rb") as model_file:
            model = pickle.load(model_file)

        MODEL_READY.set(1)

        logger.info(
            "Modelo carregado com sucesso | caminho=%s",
            MODEL_PATH,
        )

    except Exception:
        MODEL_READY.set(0)

        logger.exception(
            "Falha ao carregar o modelo | caminho=%s",
            MODEL_PATH,
        )

        raise

    yield

    logger.info("Encerrando API de diagnóstico")


# ============================================================
# Aplicação
# ============================================================

app = FastAPI(
    title="API de Predição de Diagnóstico",
    description=(
        "API responsável por receber características numéricas "
        "e executar predições utilizando um modelo de Machine Learning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Schemas
# ============================================================

class PredictionInput(BaseModel):
    """
    Dados de entrada esperados pelo modelo.
    """

    concave_points_mean: float = Field(
        ...,
        ge=0,
        description="Média dos pontos côncavos",
        examples=[0.052],
    )

    concave_points_worst: float = Field(
        ...,
        ge=0,
        description="Pior valor dos pontos côncavos",
        examples=[0.146],
    )

    perimeter_worst: float = Field(
        ...,
        gt=0,
        description="Pior valor do perímetro",
        examples=[108.4],
    )

    radius_worst: float = Field(
        ...,
        gt=0,
        description="Pior valor do raio",
        examples=[16.9],
    )


class PredictionOutput(BaseModel):
    """
    Resposta retornada pela API.
    """

    output: str
    processing_time_ms: float
    request_id: str


# ============================================================
# Middleware de monitoramento
# ============================================================

@app.middleware("http")
async def monitoring_middleware(
    request: Request,
    call_next,
):
    """
    Registra o tempo, o status e um identificador para cada requisição.
    """

    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        duration = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code="500",
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        logger.exception(
            "Erro não tratado | request_id=%s | method=%s | "
            "path=%s | duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration * 1000,
        )

        raise

    duration = time.perf_counter() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=str(response.status_code),
    ).inc()

    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.6f}"

    logger.info(
        "Requisição finalizada | request_id=%s | method=%s | "
        "path=%s | status=%s | duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration * 1000,
    )

    return response


# ============================================================
# Endpoints
# ============================================================

@app.get("/", tags=["Application"])
def read_root():
    return {
        "application": "API de Predição de Diagnóstico",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health/live", tags=["Health"])
def liveness():
    """
    Informa se o processo da API está em execução.
    """

    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
def readiness():
    """
    Informa se a API está pronta para receber predições.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não está disponível",
        )

    return {
        "status": "ready",
        "model_loaded": True,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    """
    Expõe métricas no padrão Prometheus.
    """

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
)
def make_predict(
    prediction_input: PredictionInput,
    request: Request,
):
    """
    Executa a predição utilizando o modelo carregado.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo indisponível",
        )

    start_time = time.perf_counter()
    request_id = request.state.request_id

    try:
        novo_input = [[
            prediction_input.concave_points_mean,
            prediction_input.concave_points_worst,
            prediction_input.perimeter_worst,
            prediction_input.radius_worst,
        ]]

        prediction = model.predict(novo_input)
        prediction_value = str(prediction[0])

        processing_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        PREDICTION_COUNT.labels(
            prediction=prediction_value
        ).inc()

        logger.info(
            "Predição realizada | request_id=%s | "
            "prediction=%s | processing_time_ms=%.2f",
            request_id,
            prediction_value,
            processing_time_ms,
        )

        return PredictionOutput(
            output=prediction_value,
            processing_time_ms=round(
                processing_time_ms,
                2,
            ),
            request_id=request_id,
        )

    except Exception as error:
        PREDICTION_ERRORS.labels(
            error_type=type(error).__name__
        ).inc()

        logger.exception(
            "Erro durante predição | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Erro interno durante a predição",
        ) from error
