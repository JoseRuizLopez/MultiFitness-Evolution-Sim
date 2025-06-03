import random
import copy
from typing import List, Callable


class NSGAII:
    """Implementación muy simplificada del algoritmo NSGA-II."""

    def __init__(self, population: List, fitness_fn: Callable, crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1):
        self.population = population
        self.fitness_fn = fitness_fn
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    # --- Utilidades de evaluación y ordenamiento ---
    @staticmethod
    def dominates(f1, f2):
        """Devuelve True si f1 domina a f2."""
        better_or_equal = all(a >= b for a, b in zip(f1, f2))
        strictly_better = any(a > b for a, b in zip(f1, f2))
        return better_or_equal and strictly_better

    def fast_non_dominated_sort(self, fitness_values):
        S = [[] for _ in fitness_values]
        front = [[]]
        n = [0 for _ in fitness_values]
        rank = [0 for _ in fitness_values]
        for p, fp in enumerate(fitness_values):
            for q, fq in enumerate(fitness_values):
                if self.dominates(fp, fq):
                    S[p].append(q)
                elif self.dominates(fq, fp):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                front[0].append(p)
        i = 0
        while front[i]:
            next_front = []
            for p in front[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            front.append(next_front)
        if not front[-1]:
            front.pop()
        return front, rank

    @staticmethod
    def crowding_distance(front, fitness_values):
        distance = [0.0 for _ in front]
        num_objectives = len(fitness_values[0])
        for m in range(num_objectives):
            values = [fitness_values[i][m] for i in front]
            sorted_indices = sorted(range(len(values)), key=lambda x: values[x])
            distance[sorted_indices[0]] = float('inf')
            distance[sorted_indices[-1]] = float('inf')
            min_val = values[sorted_indices[0]]
            max_val = values[sorted_indices[-1]]
            if max_val - min_val == 0:
                continue
            for j in range(1, len(front) - 1):
                prev_val = values[sorted_indices[j - 1]]
                next_val = values[sorted_indices[j + 1]]
                distance[sorted_indices[j]] += (next_val - prev_val) / (max_val - min_val)
        return distance

    def binary_tournament(self, population, ranks, distances):
        i, j = random.sample(range(len(population)), 2)
        if ranks[i] < ranks[j]:
            return population[i]
        if ranks[j] < ranks[i]:
            return population[j]
        if distances[i] > distances[j]:
            return population[i]
        return population[j]

    # --- Operadores genéticos ---
    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        g1, g2 = parent1.genotype, parent2.genotype
        point = random.randint(1, len(g1) - 1)
        child1_genotype = g1[:point] + g2[point:]
        child2_genotype = g2[:point] + g1[point:]
        return parent1.__class__(genotype=child1_genotype), parent2.__class__(genotype=child2_genotype)

    def mutate(self, individual):
        for i, gene in enumerate(individual.genotype):
            if random.random() < self.mutation_rate:
                individual.genotype[i] = gene + random.uniform(-0.1, 0.1)
        return individual

    # --- Ciclo principal de una generación ---
    def step(self):
        fitness = [self.fitness_fn(ind) for ind in self.population]
        fronts, ranks = self.fast_non_dominated_sort(fitness)
        crowding = [0.0] * len(self.population)
        for front in fronts:
            distances = self.crowding_distance(front, fitness)
            for idx, d in zip(front, distances):
                crowding[idx] = d

        # Crear descendencia
        offspring = []
        while len(offspring) < len(self.population):
            parent1 = self.binary_tournament(self.population, ranks, crowding)
            parent2 = self.binary_tournament(self.population, ranks, crowding)
            child1, child2 = self.crossover(parent1, parent2)
            offspring.append(self.mutate(child1))
            if len(offspring) < len(self.population):
                offspring.append(self.mutate(child2))

        # Evaluar descendencia
        offspring_fitness = [self.fitness_fn(ind) for ind in offspring]
        combined = self.population + offspring
        combined_fitness = fitness + offspring_fitness
        fronts, ranks = self.fast_non_dominated_sort(combined_fitness)

        # Selección ambiental
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
        self.population = new_population
        return combined_fitness


# Retrocompatibilidad con la implementación previa
GeneticAlgorithm = NSGAII
