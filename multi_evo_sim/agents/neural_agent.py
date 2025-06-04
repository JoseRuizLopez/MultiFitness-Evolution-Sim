import numpy as np

from .base_agent import Action, ActionType, BaseAgent
from .simple_network import SimpleNeuralNetwork


class NeuralAgent(BaseAgent):
    """Agente que utiliza una red neuronal simple como política."""

    def __init__(self, input_size, hidden_size=4, output_size=6, genotype=None):
        # Si se proporciona un genotipo, inicializa la red a partir de él
        network = SimpleNeuralNetwork(input_size, hidden_size, output_size, genotype)
        super().__init__(network.genotype.tolist(), color="blue")
        self.network = network

    def act(self, observation):
        """Convierte la observación en un vector numérico y decide la acción."""
        # Algunas entradas de la observación contienen secuencias (por ejemplo
        # la posición o la lista de recursos). Convertir ese diccionario en un
        # array directamente provoca un ``ValueError``. Extraemos únicamente
        # valores numéricos para alimentar a la red neuronal.
        position = observation.get("position", (0, 0))
        resource_here = observation.get("resource_here", 0)
        inventory = observation.get("inventory", 0)
        danger = 1 if observation.get("danger", False) else 0
        num_resources = len(observation.get("resources", []))

        features = np.array(
            [position[0], position[1], resource_here, inventory, num_resources, danger],
            dtype=float,
        )

        # Ajustamos el vector al tamaño que espera la red neuronal
        if self.network.input_size < features.size:
            obs_vector = features[: self.network.input_size]
        elif self.network.input_size > features.size:
            obs_vector = np.pad(features, (0, self.network.input_size - features.size))
        else:
            obs_vector = features

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
