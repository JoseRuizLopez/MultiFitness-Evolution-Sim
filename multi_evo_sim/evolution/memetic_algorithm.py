from __future__ import annotations
import copy
import random
from typing import Callable, List

from .genetic_algorithm import NSGAII


class MemeticNSGAII(NSGAII):
    """Extiende :class:`NSGAII` con una fase de búsqueda local.

    La búsqueda local aplica mutaciones pequeñas adicionales a cada
    descendiente y se queda con la variante que domine a la original
    según la función de fitness.
    """

    def __init__(
        self,
        population: List,
        fitness_fn: Callable,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        n_jobs: int = 1,
        local_search_iters: int = 3,
        local_mutation: float = 0.1,
    ) -> None:
        super().__init__(
            population,
            fitness_fn,
            crossover_rate,
            mutation_rate,
            n_jobs=n_jobs,
        )
        self.local_search_iters = local_search_iters
        self.local_mutation = local_mutation

    def local_search(self, individual):
        """Realiza un pequeño hill climbing sobre ``individual``."""
        best = copy.deepcopy(individual)
        best_fit = self.fitness_fn(best)
        for _ in range(self.local_search_iters):
            cand = copy.deepcopy(best)
            for i, gene in enumerate(cand.genotype):
                if random.random() < self.mutation_rate:
                    cand.genotype[i] = gene + random.uniform(
                        -self.local_mutation, self.local_mutation
                    )
            if hasattr(cand, "update_network"):
                cand.update_network()
            cand_fit = self.fitness_fn(cand)
            if self.dominates(cand_fit, best_fit):
                best, best_fit = cand, cand_fit
        return best

    def step(self):
        """Ejecuta un paso evolutivo con búsqueda local y elitismo."""
        fitness = self._evaluate_population(self.population)
        fronts, ranks = self.fast_non_dominated_sort(fitness)
        crowding = [0.0] * len(self.population)
        for front in fronts:
            distances = self.crowding_distance(front, fitness)
            for idx, d in zip(front, distances):
                crowding[idx] = d

        # Identificar el mejor individuo actual para garantizar elitismo
        best_idx = min(range(len(self.population)), key=lambda i: (ranks[i], -crowding[i]))
        elite = copy.deepcopy(self.population[best_idx])

        offspring = []
        while len(offspring) < len(self.population):
            parent1 = self.binary_tournament(self.population, ranks, crowding)
            parent2 = self.binary_tournament(self.population, ranks, crowding)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            child1 = self.local_search(child1)
            child2 = self.local_search(child2)
            offspring.append(child1)
            if len(offspring) < len(self.population):
                offspring.append(child2)

        offspring_fitness = self._evaluate_population(offspring)
        combined = self.population + offspring
        combined_fitness = fitness + offspring_fitness
        fronts, ranks = self.fast_non_dominated_sort(combined_fitness)

        new_population = []
        for front in fronts:
            if len(new_population) + len(front) > len(self.population):
                distances = self.crowding_distance(front, combined_fitness)
                sorted_front = [f for _, f in sorted(zip(distances, front), reverse=True)]
                for idx in sorted_front:
                    if len(new_population) < len(self.population):
                        new_population.append(combined[idx])
                    else:
                        break
                break
            else:
                new_population.extend([combined[i] for i in front])

        # Asegurar que el mejor individuo permanezca en la población
        if not any(ind.genotype == elite.genotype for ind in new_population):
            np_fitness = self._evaluate_population(new_population)
            np_fronts, np_ranks = self.fast_non_dominated_sort(np_fitness)
            np_crowding = [0.0] * len(new_population)
            for front in np_fronts:
                distances = self.crowding_distance(front, np_fitness)
                for idx, d in zip(front, distances):
                    np_crowding[idx] = d
            worst_idx = max(range(len(new_population)), key=lambda i: (np_ranks[i], -np_crowding[i]))
            new_population[worst_idx] = elite

        self.population = new_population
        return combined_fitness

# Retrocompatibilidad con la implementación previa
MemeeticAlgorithm = MemeticNSGAII