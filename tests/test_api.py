"""Testes da API FastAPI de predição."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_retorna_modelo_carregado(client):
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["model_loaded"] is True
    assert corpo["model_file"] in {
        "modelo_pipeline_genetico.pkl",
        "modelo_pipeline.pkl",
    }


def test_raiz_informa_endpoints(client):
    resposta = client.get("/")
    assert resposta.status_code == 200
    assert "predict" in resposta.json()


def test_predict_retorna_classe(client):
    resposta = client.get(
        "/predict",
        params={
            "concave_points_mean": 0.052,
            "concave_points_worst": 0.146,
            "perimeter_worst": 108.4,
            "radius_worst": 16.9,
        },
    )
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert "output" in corpo
    assert corpo["output"] in {"0", "1", "0.0", "1.0", 0, 1} or str(corpo["output"]) in {
        "0",
        "1",
    }
    assert "X-Process-Time-Ms" in resposta.headers


def test_predict_rejeita_valor_negativo(client):
    resposta = client.get(
        "/predict",
        params={
            "concave_points_mean": -1,
            "concave_points_worst": 0.1,
            "perimeter_worst": 100,
            "radius_worst": 10,
        },
    )
    assert resposta.status_code == 422


def test_arquivos_de_modelo_existem():
    raiz = Path(__file__).resolve().parents[1]
    assert (raiz / "modelo_pipeline.pkl").exists() or (
        raiz / "modelo_pipeline_genetico.pkl"
    ).exists()
