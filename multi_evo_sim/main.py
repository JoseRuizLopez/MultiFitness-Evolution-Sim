from .env.world import World
from .env.resource import Resource
from .agents.neural_agent import NeuralAgent
from .agents.base_agent import BaseAgent
from .config import POPULATION_SIZE
from .evolution.genetic_algorithm import GeneticAlgorithm
from .evolution.fitness_functions import fitness_combinado
from .visualization.logger import ExperimentLogger, log
from .visualization.render import Renderer
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

    renderer = Renderer()
    exp_logger = ExperimentLogger()

    # O alternativamente: población evolutiva
    population = [random_agent() for _ in range(POPULATION_SIZE)]
    for ag in population:
        world.add_agent(ag, position=(0, 0))

    ga = GeneticAlgorithm(population, fitness_combinado)
    fitness = ga.step()
    fronts, _ = ga.fast_non_dominated_sort(fitness)
    exp_logger.log_fitness(0, fitness)
    if fronts:
        exp_logger.log_pareto_front(0, fronts[0], fitness)
    log(f"Fitness calculado: {fitness}")

    # Ejecutar un paso en el mundo para que los agentes actúen
    for step in range(3):
        world.step()
        renderer.draw(world)
    for ag in population:
        log(f"Inventario del agente: {ag.inventory}")
    log(f"Inventario del agente individual (Neural): {agent.inventory}")
    exp_logger.save()


if __name__ == "__main__":
    run_simulation()
