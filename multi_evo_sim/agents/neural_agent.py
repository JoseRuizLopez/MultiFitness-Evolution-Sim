import numpy as np
import logging

from .base_agent import Action, ActionType, BaseAgent
from .simple_network import SimpleNeuralNetwork

logger = logging.getLogger(__name__)


class NeuralAgent(BaseAgent):
    """Agente que utiliza una red neuronal simple como política."""

    def __init__(self, input_size=None, hidden_size=4, output_size=6, genotype=None):
        """Crea un ``NeuralAgent`` con una red neuronal mínima.

        Si ``input_size`` es ``None`` se calcula automáticamente a partir del
        número de características que genera :meth:`_extract_features`.
        """
        if input_size is None:
            input_size = self.feature_size()

        network = SimpleNeuralNetwork(input_size, hidden_size, output_size, genotype)
        super().__init__(network.genotype.tolist(), color="blue")
        self.network = network

    def update_network(self):
        """Rebuild ``self.network`` from ``self.genotype``."""
        self.network = SimpleNeuralNetwork(
            self.network.input_size,
            self.network.hidden_size,
            self.network.output_size,
            self.genotype,
        )

    @staticmethod
    def _extract_features(observation):
        """Devuelve un ``numpy.ndarray`` con las características numéricas de la observación."""
        position = observation.get("position", (0, 0))
        resources = observation.get("resources", [])
        resource_here = int(bool(observation.get("resource_here", 0)))
        inventory = observation.get("inventory", 0)
        danger = 1 if observation.get("danger", False) else 0
        num_resources = len(resources)

        if resources:
            nearest = min(
                resources,
                key=lambda r: (r[0] - position[0]) ** 2 + (r[1] - position[1]) ** 2,
            )
            dx = nearest[0] - position[0]
            dy = nearest[1] - position[1]
        else:
            dx = 0
            dy = 0

        return np.array(
            [
                position[0],
                position[1],
                dx,
                dy,
                resource_here,
                inventory,
                num_resources,
                danger,
            ],
            dtype=float,
        )

    @classmethod
    def feature_size(cls):
        """Número de valores que produce :meth:`_extract_features`."""
        return cls._extract_features({}).size

    def act(self, observation):
        """Convierte la observación en un vector numérico y decide la acción."""
        # Algunas entradas de la observación contienen secuencias (por ejemplo
        # la posición o la lista de recursos). Convertir ese diccionario en un
        # array directamente provoca un ``ValueError``. Por ello se utiliza la
        # función auxiliar ``_extract_features`` que devuelve únicamente valores
        # numéricos.
        features = self._extract_features(observation)

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
            logger.info(
                "GATHER action triggered at %s", observation.get("position")
            )
            return Action(ActionType.GATHER)
        return Action(ActionType.COOPERATE)
