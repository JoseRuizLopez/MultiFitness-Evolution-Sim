from .env.world import World
from .env.resource import Resource
from .agents.neural_agent import NeuralAgent
from .agents.base_agent import BaseAgent
from .config import POPULATION_SIZE
from .evolution.genetic_algorithm import GeneticAlgorithm
from .evolution.memetic_algorithm import MemeticAlgorithm
from .evolution.fitness_functions import fitness_combinado
from .visualization.logger import ExperimentLogger, log
from .visualization.render import Renderer
import random
import argparse


def random_agent(size=5):
    genotype = [random.random() for _ in range(size)]
    return BaseAgent(genotype=genotype)


def run_simulation(record: bool = False, video_path: str = "sim.mp4"):
    renderer = Renderer(record=record, video_path=video_path)
    exp_logger = ExperimentLogger()

    # Mundo con un recurso inicial en (0, 0)
    world = World(width=10, height=10, resources=[Resource((0, 0))])
    renderer.draw(world, generation=0)

    # Agente individual para prueba rápida (NeuralAgent)
    agent = NeuralAgent()
    world.add_agent(agent, position=(0, 0))
    renderer.draw(world, generation=0)

    # O alternativamente: población evolutiva
    population = [random_agent() for _ in range(POPULATION_SIZE)]
    for ag in population:
        pos = (random.randint(0, world.width - 1), random.randint(0, world.height - 1))
        world.add_agent(ag, position=pos)
        renderer.draw(world, generation=0)

    ga = MemeticAlgorithm(population, fitness_combinado)
    fitness = ga.step()
    fronts, _ = ga.fast_non_dominated_sort(fitness)
    exp_logger.log_fitness(0, fitness)
    if fronts:
        exp_logger.log_pareto_front(0, fronts[0], fitness)
    log(f"Fitness calculado: {fitness}")

    # Ejecutar un paso en el mundo para que los agentes actúen
    for step in range(200):
        world.step()
        renderer.draw(world, generation=0)
    for ag in population:
        log(f"Inventario del agente: {ag.inventory}")
    log(f"Inventario del agente individual (Neural): {agent.inventory}")
    exp_logger.save()
    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="store_true",
        help="Guardar un video de la simulación",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="sim.mp4",
        help="Ruta del archivo de video a generar",
    )
    args = parser.parse_args()
    run_simulation(record=args.record, video_path=args.video_path)
