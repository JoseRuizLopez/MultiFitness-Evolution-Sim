import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multi_evo_sim.agents.base_agent import BaseAgent
from multi_evo_sim.evolution.genetic_algorithm import NSGAII


def test_dominates():
    assert NSGAII.dominates([2, 2], [1, 1]) is True
    assert NSGAII.dominates([1, 1], [2, 2]) is False
    assert NSGAII.dominates([1, 2], [1, 1]) is True
    assert NSGAII.dominates([1, 1], [1, 1]) is False


def test_nsga_step_population_size_constant():
    population = [BaseAgent(genotype=[i, i]) for i in range(4)]
    ga = NSGAII(population, lambda ag: [len(ag.genotype)])
    ga.step()
    assert len(ga.population) == 4
