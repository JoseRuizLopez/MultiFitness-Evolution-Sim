from .env.world import World
from .env.resource import Resource
from .agents.neural_agent import NeuralAgent
from .evolution.genetic_algorithm import GeneticAlgorithm
from .evolution.fitness_functions import example_fitness
from .visualization.logger import log


def run_simulation():
    # Mundo con un recurso inicial en (0, 0)
    world = World(width=10, height=10, resources=[Resource((0, 0))])

    agent = NeuralAgent(input_size=2)
    world.add_agent(agent, position=(0, 0))

    ga = GeneticAlgorithm([agent], example_fitness)
    fitness = ga.step()
    log(f"Fitness calculado: {fitness}")

    # Ejecutar un paso en el mundo para que el agente actúe
    world.step()
    log(f"Inventario del agente: {agent.inventory}")


if __name__ == "__main__":
    run_simulation()
