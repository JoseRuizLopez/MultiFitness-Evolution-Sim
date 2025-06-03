class BaseAgent:
    """Agente base del sistema evolutivo."""

    def __init__(self, genotype=None):
        self.genotype = genotype or []

        # Métricas básicas utilizadas por las funciones de fitness
        self.resources_collected = 0
        self.shared_resources = 0
        self.alive = True
        self.steps_survived = 0

    def act(self, observation):
        """Define la acción del agente dado un estado de observación."""
        raise NotImplementedError
