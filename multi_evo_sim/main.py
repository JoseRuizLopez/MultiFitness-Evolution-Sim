from .env.world import World
from .agents.base_agent import BaseAgent
from .config import POPULATION_SIZE
from .evolution.genetic_algorithm import GeneticAlgorithm
from .evolution.fitness_functions import example_fitness
from .visualization.logger import log
import random


def random_agent(size=5):
    genotype = [random.random() for _ in range(size)]
    return BaseAgent(genotype=genotype)


def run_simulation():
    world = World(width=10, height=10)
    population = [random_agent() for _ in range(POPULATION_SIZE)]
    for ag in population:
        world.add_agent(ag, position=(0, 0))

    ga = GeneticAlgorithm(population, example_fitness)
    fitness = ga.step()
    log(f"Fitness calculado: {fitness}")


if __name__ == "__main__":
    run_simulation()
