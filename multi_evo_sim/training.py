import random

from .env.world import World
from .env.resource import Resource
from .agents.base_agent import BaseAgent
from .agents.neural_agent import NeuralAgent
from .evolution.genetic_algorithm import GeneticAlgorithm
from .evolution.memetic_algorithm import MemeeticAlgorithm
from .evolution.fitness_functions import fitness_combinado
from .visualization.logger import ExperimentLogger, log
from .visualization.render import Renderer


renderer = Renderer()

def _evaluate_agent(agent, steps: int = 100, draw: bool=False) -> list:
    """Ejecuta una simulaci\u00f3n corta con compa\u00f1eros y devuelve el fitness."""
    agent.inventory = 0
    agent.resources_collected = 0
    agent.shared_resources = 0
    agent.alive = True
    agent.steps_survived = 0 

    # Mas recursos y un agente adicional permiten que se observe cooperacon y
    # crecimiento durante la evaluacion, generando fitness mas variado.
    resources = [Resource((random.randint(0, 9), random.randint(0, 9))) for _ in range(10)]
    world = World(width=10, height=10, resources=resources)
    world.add_agent(agent, position=(random.randint(0, 9), random.randint(0, 9)))
    # Agregar un compañero basico para posibilitar acciones de cooperacion
    world.add_agent(BaseAgent(), position=(random.randint(0, 9), random.randint(0, 9)))
    for _ in range(steps):
        world.step()
        if draw:
            renderer.draw(world)
    return fitness_combinado(agent)


def train(population_size=10, generations=1000):
    """Lanza un proceso evolutivo sencillo con NSGA-II."""
    population = [NeuralAgent() for _ in range(population_size)]
    ga = MemeeticAlgorithm(population, _evaluate_agent)
    logger = ExperimentLogger()

    for gen in range(1, generations+1):
        if gen % 500 == 0:
            fitness = [_evaluate_agent(ind, draw=True) for ind in ga.population]
        else:
            fitness = [_evaluate_agent(ind, draw=False) for ind in ga.population]
        fronts, _ = ga.fast_non_dominated_sort(fitness)
        logger.log_fitness(gen, fitness)
        inventories = [ind.inventory for ind in ga.population]
        logger.log_inventory(gen, inventories)
        if fronts:
            logger.log_pareto_front(gen, fronts[0], fitness)
        log(f"Generaci\u00f3n {gen} fitness: {fitness}")
        ga.step()

    logger.save()


if __name__ == "__main__":
    train()
