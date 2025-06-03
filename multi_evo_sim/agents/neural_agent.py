from .base_agent import BaseAgent


class NeuralAgent(BaseAgent):
    """Agente que utiliza una red neuronal simple como política."""

    def __init__(self, network, genotype=None):
        super().__init__(genotype)
        self.network = network

    def act(self, observation):
        """Aplica la red neuronal al vector de observación."""
        return self.network.forward(observation)
