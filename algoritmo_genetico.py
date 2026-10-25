"""Algoritmo genético para otimização de hiperparâmetros de classificadores."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score


def criar_individuo(espaco_busca: dict[str, list[Any]]) -> dict[str, Any]:
    """Cria uma combinação aleatória de hiperparâmetros."""
    return {
        parametro: random.choice(valores)
        for parametro, valores in espaco_busca.items()
    }


def criar_populacao(
    espaco_busca: dict[str, list[Any]],
    tamanho_populacao: int,
) -> list[dict[str, Any]]:
    """Cria a população inicial do algoritmo genético."""
    return [criar_individuo(espaco_busca) for _ in range(tamanho_populacao)]


def chave_individuo(individuo: dict[str, Any]) -> tuple:
    """Transforma o indivíduo em chave para reutilizar avaliações anteriores."""
    return tuple(sorted(individuo.items()))


def avaliar_individuo(
    individuo: dict[str, Any],
    fabrica_modelo: Callable[[dict[str, Any]], Any],
    x_treino,
    y_treino,
    validacao,
    cache: dict,
    scoring: str = "recall",
) -> float:
    """Calcula o score médio do indivíduo com validação cruzada."""
    chave = chave_individuo(individuo)

    if chave not in cache:
        modelo = fabrica_modelo(individuo)
        resultados = cross_val_score(
            modelo,
            x_treino,
            y_treino,
            cv=validacao,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        cache[chave] = float(resultados.mean())

    return cache[chave]


def selecionar_por_torneio(
    populacao_avaliada: list[tuple[dict[str, Any], float]],
    tamanho_torneio: int,
) -> dict[str, Any]:
    """Seleciona o melhor indivíduo de uma amostra aleatória da população."""
    competidores = random.sample(
        populacao_avaliada,
        k=min(tamanho_torneio, len(populacao_avaliada)),
    )
    vencedor = max(competidores, key=lambda item: item[1])
    return deepcopy(vencedor[0])


def aplicar_crossover(
    pai_1: dict[str, Any],
    pai_2: dict[str, Any],
    taxa_crossover: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Combina os hiperparâmetros dos pais com crossover uniforme."""
    if random.random() >= taxa_crossover:
        return deepcopy(pai_1), deepcopy(pai_2)

    filho_1: dict[str, Any] = {}
    filho_2: dict[str, Any] = {}

    for parametro in pai_1:
        if random.random() < 0.5:
            filho_1[parametro] = pai_1[parametro]
            filho_2[parametro] = pai_2[parametro]
        else:
            filho_1[parametro] = pai_2[parametro]
            filho_2[parametro] = pai_1[parametro]

    return filho_1, filho_2


def aplicar_mutacao(
    individuo: dict[str, Any],
    espaco_busca: dict[str, list[Any]],
    taxa_mutacao: float,
) -> dict[str, Any]:
    """Altera genes do indivíduo de acordo com a taxa de mutação."""
    individuo_mutado = deepcopy(individuo)

    for parametro, valores in espaco_busca.items():
        if random.random() < taxa_mutacao:
            individuo_mutado[parametro] = random.choice(valores)

    return individuo_mutado


def executar_algoritmo_genetico(
    fabrica_modelo: Callable[[dict[str, Any]], Any],
    espaco_busca: dict[str, list[Any]],
    x_treino,
    y_treino,
    tamanho_populacao: int = 12,
    numero_geracoes: int = 8,
    taxa_crossover: float = 0.8,
    taxa_mutacao: float = 0.15,
    tamanho_torneio: int = 3,
    quantidade_elites: int = 2,
    scoring: str = "recall",
    semente: int = 42,
    verbose: bool = True,
) -> tuple[dict[str, Any], float, list[dict[str, float]]]:
    """Executa a busca genética e retorna o melhor indivíduo encontrado."""
    random.seed(semente)
    np.random.seed(semente)

    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
    cache_avaliacoes: dict = {}
    populacao = criar_populacao(espaco_busca, tamanho_populacao)
    historico: list[dict[str, float]] = []

    for geracao in range(1, numero_geracoes + 1):
        populacao_avaliada = [
            (
                individuo,
                avaliar_individuo(
                    individuo,
                    fabrica_modelo,
                    x_treino,
                    y_treino,
                    validacao,
                    cache_avaliacoes,
                    scoring=scoring,
                ),
            )
            for individuo in populacao
        ]
        populacao_avaliada.sort(key=lambda item: item[1], reverse=True)

        melhor_individuo, melhor_fitness = populacao_avaliada[0]
        fitness_medio = float(np.mean([fitness for _, fitness in populacao_avaliada]))
        historico.append(
            {
                "geracao": geracao,
                "melhor_fitness": melhor_fitness,
                "fitness_medio": fitness_medio,
            }
        )

        if verbose:
            print(
                f"Geracao {geracao:02d} | "
                f"Melhor {scoring}: {melhor_fitness:.4f} | "
                f"{scoring} medio: {fitness_medio:.4f}"
            )

        nova_populacao = [
            deepcopy(individuo)
            for individuo, _ in populacao_avaliada[:quantidade_elites]
        ]

        while len(nova_populacao) < tamanho_populacao:
            pai_1 = selecionar_por_torneio(populacao_avaliada, tamanho_torneio)
            pai_2 = selecionar_por_torneio(populacao_avaliada, tamanho_torneio)
            filho_1, filho_2 = aplicar_crossover(pai_1, pai_2, taxa_crossover)
            filho_1 = aplicar_mutacao(filho_1, espaco_busca, taxa_mutacao)
            filho_2 = aplicar_mutacao(filho_2, espaco_busca, taxa_mutacao)
            nova_populacao.extend([filho_1, filho_2])

        populacao = nova_populacao[:tamanho_populacao]

    populacao_final = [
        (
            individuo,
            avaliar_individuo(
                individuo,
                fabrica_modelo,
                x_treino,
                y_treino,
                validacao,
                cache_avaliacoes,
                scoring=scoring,
            ),
        )
        for individuo in populacao
    ]
    melhor_individuo, melhor_fitness = max(populacao_final, key=lambda item: item[1])
    return deepcopy(melhor_individuo), melhor_fitness, historico


# Três configurações distintas exigidas pelo Tech Challenge
EXPERIMENTOS_AG = [
    {
        "nome": "Conservador",
        "descricao": "População menor, poucas gerações e baixa mutação (explora pouco).",
        "tamanho_populacao": 8,
        "numero_geracoes": 10,
        "taxa_crossover": 0.7,
        "taxa_mutacao": 0.05,
        "tamanho_torneio": 2,
        "quantidade_elites": 1,
    },
    {
        "nome": "Equilibrado",
        "descricao": "Configuração intermediária entre exploração e explotação.",
        "tamanho_populacao": 12,
        "numero_geracoes": 10,
        "taxa_crossover": 0.8,
        "taxa_mutacao": 0.15,
        "tamanho_torneio": 3,
        "quantidade_elites": 2,
    },
    {
        "nome": "Exploratorio",
        "descricao": "População maior e mutação alta para diversificar a busca.",
        "tamanho_populacao": 20,
        "numero_geracoes": 10,
        "taxa_crossover": 0.9,
        "taxa_mutacao": 0.30,
        "tamanho_torneio": 4,
        "quantidade_elites": 2,
    },
]
