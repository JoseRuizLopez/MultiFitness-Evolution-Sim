class BaseAgent:
    """Agente base del sistema evolutivo."""

    def __init__(self, genotype=None):
        self.genotype = genotype or []

    def act(self, observation):
        """Define la acción del agente dado un estado de observación."""
        raise NotImplementedError
