import random

from .env.world import World
from .env.resource import Resource
from .agents.neural_agent import NeuralAgent
from .evolution.genetic_algorithm import GeneticAlgorithm
from .evolution.fitness_functions import fitness_combinado
from .visualization.logger import ExperimentLogger, log


def _evaluate_agent(agent, steps=20):
    """Ejecuta una simulaci\u00f3n corta y devuelve el fitness."""
    agent.inventory = 0
    agent.resources_collected = 0
    agent.shared_resources = 0
    agent.alive = True
    agent.steps_survived = 0

    resources = [Resource((random.randint(0, 9), random.randint(0, 9))) for _ in range(3)]
    world = World(width=10, height=10, resources=resources)
    world.add_agent(agent, position=(random.randint(0, 9), random.randint(0, 9)))
    for _ in range(steps):
        world.step()
    return fitness_combinado(agent)


def train(population_size=10, generations=5):
    """Lanza un proceso evolutivo sencillo con NSGA-II."""
    population = [NeuralAgent() for _ in range(population_size)]
    ga = GeneticAlgorithm(population, _evaluate_agent)
    logger = ExperimentLogger()

    for gen in range(generations):
        fitness = [_evaluate_agent(ind) for ind in ga.population]
        fronts, _ = ga.fast_non_dominated_sort(fitness)
        logger.log_fitness(gen, fitness)
        if fronts:
            logger.log_pareto_front(gen, fronts[0], fitness)
        log(f"Generaci\u00f3n {gen} fitness: {fitness}")
        ga.step()

    logger.save()


if __name__ == "__main__":
    train()
