"""
NSGA-II corrigido — trabalha com índices inteiros, sem bug de hashability.
"""
from __future__ import annotations
import numpy as np
from dnaty.core.individual import Individual


def fast_non_dominated_sort(fitnesses: list[tuple]) -> list[list[int]]:
    """
    Input:  lista de tuples de fitness (todos a maximizar)
    Output: lista de fronts, cada front é lista de ÍNDICES inteiros
    """
    n = len(fitnesses)
    domination_count = [0] * n
    dominated_by = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            fi, fj = fitnesses[i], fitnesses[j]
            i_dom_j = all(a >= b for a, b in zip(fi, fj)) and any(a > b for a, b in zip(fi, fj))
            j_dom_i = all(b >= a for a, b in zip(fi, fj)) and any(b > a for a, b in zip(fi, fj))
            if i_dom_j:
                dominated_by[i].append(j)
            elif j_dom_i:
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
        current += 1

    return [f for f in fronts if f]


def crowding_distance(fitnesses: list[tuple], front: list[int]) -> dict[int, float]:
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    n_obj = len(fitnesses[0])
    cd: dict[int, float] = {i: 0.0 for i in front}
    for obj in range(n_obj):
        sorted_front = sorted(front, key=lambda i: fitnesses[i][obj])
        cd[sorted_front[0]] = float("inf")
        cd[sorted_front[-1]] = float("inf")
        f_min = fitnesses[sorted_front[0]][obj]
        f_max = fitnesses[sorted_front[-1]][obj]
        if f_max == f_min:
            continue
        for k in range(1, len(sorted_front) - 1):
            cd[sorted_front[k]] += (
                fitnesses[sorted_front[k + 1]][obj] - fitnesses[sorted_front[k - 1]][obj]
            ) / (f_max - f_min)
    return cd


def nsga2_select(
    population: list[Individual],
    fitnesses: list[tuple],
    n_select: int,
) -> tuple[list[Individual], list[tuple]]:
    """Seleciona n_select indivíduos por dominância + crowding distance."""
    fronts = fast_non_dominated_sort(fitnesses)
    selected_idx: list[int] = []
    for front in fronts:
        if len(selected_idx) + len(front) <= n_select:
            selected_idx.extend(front)
        else:
            cd = crowding_distance(fitnesses, front)
            sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
            needed = n_select - len(selected_idx)
            selected_idx.extend(sorted_front[:needed])
            break
    sel_pop = [population[i] for i in selected_idx]
    sel_fit = [fitnesses[i] for i in selected_idx]
    return sel_pop, sel_fit
