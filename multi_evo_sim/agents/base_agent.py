from enum import Enum


class ActionType(Enum):
    MOVE = "move"
    GATHER = "gather"
    COOPERATE = "cooperate"


class BaseAgent:
    """Agente base del sistema evolutivo."""

    def __init__(self, genotype=None):
        self.genotype = genotype or []
        # Cantidad de recursos recolectados por el agente
        self.inventory = 0

        # Métricas básicas utilizadas por las funciones de fitness
        self.resources_collected = 0
        self.shared_resources = 0
        self.alive = True
        self.steps_survived = 0

    def act(self, observation):
        """Define la acción del agente dado un estado de observación."""
        raise NotImplementedError


class Action:
    """Representa una acción ejecutada por un agente."""

    def __init__(self, action_type, **kwargs):
        self.type = ActionType(action_type)
        self.params = kwargs
