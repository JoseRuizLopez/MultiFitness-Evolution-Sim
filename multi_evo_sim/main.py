from .env.world import World
from .agents.base_agent import BaseAgent
from .evolution.genetic_algorithm import GeneticAlgorithm
from .evolution.fitness_functions import example_fitness
from .visualization.logger import log


def run_simulation():
    world = World(width=10, height=10)
    agent = BaseAgent()
    world.add_agent(agent, position=(0, 0))

    ga = GeneticAlgorithm([agent], example_fitness)
    fitness = ga.step()
    log(f"Fitness calculado: {fitness}")


if __name__ == "__main__":
    run_simulation()
