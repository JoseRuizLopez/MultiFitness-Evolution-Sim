import numpy as np

from .base_agent import Action, ActionType, BaseAgent
from .simple_network import SimpleNeuralNetwork


class NeuralAgent(BaseAgent):
    """Agente que utiliza una red neuronal simple como política."""

    def __init__(self, input_size, hidden_size=4, output_size=6, genotype=None):
        # Si se proporciona un genotipo, inicializa la red a partir de él
        network = SimpleNeuralNetwork(input_size, hidden_size, output_size, genotype)
        super().__init__(network.genotype.tolist())
        self.network = network

    def act(self, observation):
        """Convierte la salida de la red en una acción básica."""
        obs_vector = np.array(list(observation.values()), dtype=float)
        output = self.network.forward(obs_vector)
        action_idx = int(np.argmax(output))

        if action_idx == 0:
            return Action(ActionType.MOVE, direction=(1, 0))
        if action_idx == 1:
            return Action(ActionType.MOVE, direction=(-1, 0))
        if action_idx == 2:
            return Action(ActionType.MOVE, direction=(0, 1))
        if action_idx == 3:
            return Action(ActionType.MOVE, direction=(0, -1))
        if action_idx == 4:
            return Action(ActionType.GATHER)
        return Action(ActionType.COOPERATE)
