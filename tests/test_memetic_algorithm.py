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
