from .env.world import World
from .env.resource import Resource
from .agents.neural_agent import NeuralAgent
from .agents.base_agent import BaseAgent
from .config import POPULATION_SIZE
from .evolution.genetic_algorithm import GeneticAlgorithm
from .evolution.fitness_functions import fitness_combinado
from .visualization.logger import log
import random


def random_agent(size=5):
    genotype = [random.random() for _ in range(size)]
    return BaseAgent(genotype=genotype)


def run_simulation():
    # Mundo con un recurso inicial en (0, 0)
    world = World(width=10, height=10, resources=[Resource((0, 0))])

    # Agente individual para prueba rápida (NeuralAgent)
    agent = NeuralAgent(input_size=2)
    world.add_agent(agent, position=(0, 0))

    # O alternativamente: población evolutiva
    population = [random_agent() for _ in range(POPULATION_SIZE)]
    for ag in population:
        world.add_agent(ag, position=(0, 0))

    ga = GeneticAlgorithm(population, fitness_combinado)
    fitness = ga.step()
    log(f"Fitness calculado: {fitness}")

    # Ejecutar un paso en el mundo para que los agentes actúen
    world.step()
    for ag in population:
        log(f"Inventario del agente: {ag.inventory}")
    log(f"Inventario del agente individual (Neural): {agent.inventory}")


if __name__ == "__main__":
    run_simulation()
