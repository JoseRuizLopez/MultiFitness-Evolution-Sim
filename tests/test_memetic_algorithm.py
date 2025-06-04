import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multi_evo_sim.agents.base_agent import BaseAgent
from multi_evo_sim.evolution.memetic_algorithm import MemeticNSGAII


def test_memetic_nsga_step_population_size_constant():
    population = [BaseAgent(genotype=[i, i]) for i in range(4)]
    ga = MemeticNSGAII(population, lambda ag: [len(ag.genotype)])
    ga.step()
    assert len(ga.population) == 4


def test_memetic_elitism_keeps_best_individual():
    import random

    random.seed(0)
    population = [BaseAgent(genotype=[i, i]) for i in range(4)]
    fitness = lambda ag: [ag.genotype[0] + ag.genotype[1]]
    ga = MemeticNSGAII(population, fitness, mutation_rate=1.0, local_search_iters=0)
    best = max(population, key=lambda ag: ag.genotype[0] + ag.genotype[1]).genotype[:]
    ga.step()
    assert any(ind.genotype == best for ind in ga.population)


def test_memetic_elitism_persists_over_generations():
    import random

    random.seed(0)
    population = [BaseAgent(genotype=[i, i]) for i in range(4)]
    fitness = lambda ag: [ag.genotype[0] + ag.genotype[1]]
    ga = MemeticNSGAII(population, fitness, mutation_rate=0.0, local_search_iters=0)
    best = max(population, key=lambda ag: ag.genotype[0] + ag.genotype[1]).genotype[:]
    for _ in range(5):
        ga.step()
    assert any(ind.genotype == best for ind in ga.population)
