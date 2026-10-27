"""Testes do algoritmo genético e dos operadores genéticos."""

from __future__ import annotations

import random

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

from algoritmo_genetico import (
    EXPERIMENTOS_AG,
    aplicar_crossover,
    aplicar_mutacao,
    criar_individuo,
    criar_populacao,
    executar_algoritmo_genetico,
    selecionar_por_torneio,
)


@pytest.fixture
def espaco_busca():
    return {
        "n_neighbors": [1, 3, 5],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }


def test_existem_pelo_menos_tres_experimentos():
    assert len(EXPERIMENTOS_AG) >= 3
    nomes = {exp["nome"] for exp in EXPERIMENTOS_AG}
    assert len(nomes) == len(EXPERIMENTOS_AG)


def test_criar_individuo_respeita_espaco(espaco_busca):
    random.seed(0)
    individuo = criar_individuo(espaco_busca)
    assert set(individuo) == set(espaco_busca)
    for parametro, valor in individuo.items():
        assert valor in espaco_busca[parametro]


def test_criar_populacao_tamanho(espaco_busca):
    populacao = criar_populacao(espaco_busca, tamanho_populacao=5)
    assert len(populacao) == 5


def test_crossover_preserva_genes_validos(espaco_busca):
    random.seed(1)
    pai_1 = {"n_neighbors": 1, "weights": "uniform", "p": 1}
    pai_2 = {"n_neighbors": 5, "weights": "distance", "p": 2}
    filho_1, filho_2 = aplicar_crossover(pai_1, pai_2, taxa_crossover=1.0)

    for filho in (filho_1, filho_2):
        for parametro, valor in filho.items():
            assert valor in espaco_busca[parametro]


def test_mutacao_pode_alterar_gene(espaco_busca):
    random.seed(2)
    original = {"n_neighbors": 1, "weights": "uniform", "p": 1}
    mutado = aplicar_mutacao(original, espaco_busca, taxa_mutacao=1.0)
    assert set(mutado) == set(original)
    for parametro, valor in mutado.items():
        assert valor in espaco_busca[parametro]


def test_selecao_torneio_escolhe_melhor():
    populacao = [
        ({"a": 1}, 0.1),
        ({"a": 2}, 0.9),
        ({"a": 3}, 0.2),
    ]
    random.seed(3)
    selecionado = selecionar_por_torneio(populacao, tamanho_torneio=3)
    assert selecionado == {"a": 2}


def test_executar_algoritmo_genetico_melhora_ou_mantem(espaco_busca):
    x, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    def fabrica(parametros):
        return KNeighborsClassifier(**parametros)

    melhor, fitness, historico = executar_algoritmo_genetico(
        fabrica_modelo=fabrica,
        espaco_busca=espaco_busca,
        x_treino=x,
        y_treino=y,
        tamanho_populacao=6,
        numero_geracoes=3,
        taxa_crossover=0.8,
        taxa_mutacao=0.2,
        tamanho_torneio=2,
        quantidade_elites=1,
        scoring="accuracy",
        semente=42,
        verbose=False,
    )

    assert isinstance(melhor, dict)
    assert 0.0 <= fitness <= 1.0
    assert len(historico) == 3
    assert historico[-1]["melhor_fitness"] >= historico[0]["melhor_fitness"] - 1e-9
