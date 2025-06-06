import random

import argparse
import numpy as np

from .env.world import World
from .env.resource import Resource
from .agents.base_agent import BaseAgent
from .agents.neural_agent import NeuralAgent
from .evolution.genetic_algorithm import GeneticAlgorithm
from .evolution.memetic_algorithm import MemeticNSGAII
from functools import partial
import multiprocessing
from .utils.process_pool import get_pool
from .evolution.fitness_functions import fitness_combinado
from .visualization.logger import ExperimentLogger, log
from .visualization.render import Renderer
from . import config


# Renderer instance created lazily to avoid opening windows in worker processes
renderer = None
# Parámetros de grabación
_record = False
_video_path = "sim.mp4"

def _get_renderer():
    global renderer
    if renderer is None:
        renderer = Renderer(record=_record, video_path=_video_path)
    return renderer

def _evaluate_agent(
    agent,
    steps: int = 100,
    draw: bool = False,
    generation: int | None = None,
) -> list:
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
    for _ in range(3):
        world.spawn_danger_zone()
    world.add_agent(agent, position=(random.randint(0, 9), random.randint(0, 9)))
    # Agregar un compañero basico para posibilitar acciones de cooperacion
    world.add_agent(BaseAgent(), position=(random.randint(0, 9), random.randint(0, 9)))
    for _ in range(steps):
        world.step()
        if draw:
            _get_renderer().draw(world, generation)
    return fitness_combinado(agent)


def _evaluate_population(
    population,
    steps: int = 100,
    draw: bool = False,
    n_jobs: int = 1,
    generation: int | None = None,
):
    if n_jobs <= 1:
        return [
            _evaluate_agent(ind, steps=steps, draw=draw, generation=generation)
            for ind in population
        ]
    pool = get_pool(n_jobs)
    func = partial(_evaluate_agent, steps=steps, draw=draw, generation=generation)
    return list(pool.map(func, population))


def train(
    population_size: int = 2,
    generations: int = 20,
    memetic: bool = config.USE_MEMETIC_ALGORITHM,
    best_path: str = "best_genotype.npy",
):
    """Lanza un proceso evolutivo con NSGA-II o MemeticNSGAII.

    Además de registrar estadísticas en CSV, al finalizar guarda el
    genotipo del individuo con mejor fitness en ``best_path`` para
    poder reutilizarlo posteriormente.
    """
    population = [NeuralAgent() for _ in range(population_size)]
    ga_cls = MemeticNSGAII if memetic else GeneticAlgorithm
    n_jobs = multiprocessing.cpu_count()
    ga = ga_cls(population, _evaluate_agent, n_jobs=n_jobs)
    logger = ExperimentLogger()

    for gen in range(1, generations + 1):
        if gen % 10 == 0:
            fitness = _evaluate_population(
                ga.population, draw=True, n_jobs=1, generation=gen
            )
        else:
            fitness = _evaluate_population(
                ga.population, draw=False, n_jobs=ga.n_jobs, generation=gen
            )
        fronts, _ = ga.fast_non_dominated_sort(fitness)
        logger.log_fitness(gen, fitness)
        inventories = [ind.inventory for ind in ga.population]
        logger.log_inventory(gen, inventories)
        if fronts:
            logger.log_pareto_front(gen, fronts[0], fitness)
        log(f"Generaci\u00f3n {gen} fitness: {fitness}")
        ga.step()

    logger.save()

    if renderer is not None:
        renderer.close()

    if 'fitness' in locals() and fitness:
        best_idx = max(
            range(len(fitness)),
            key=lambda i: fitness[i][0] if isinstance(fitness[i], (list, tuple)) else fitness[i],
        )
        np.save(best_path, ga.population[best_idx].genotype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memetic",
        action="store_true",
        help="Usar MemeticNSGAII en lugar de GeneticAlgorithm",
    )
    parser.add_argument(
        "--best-path",
        type=str,
        default="best_genotype.npy",
        help="Ruta para guardar el genotipo con mejor fitness",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Guardar un video de la simulación de evaluación",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="sim.mp4",
        help="Ruta del archivo de video a generar",
    )
    args = parser.parse_args()
    _record = args.record
    _video_path = args.video_path
    train(
        memetic=args.memetic or config.USE_MEMETIC_ALGORITHM,
        best_path=args.best_path,
    )
