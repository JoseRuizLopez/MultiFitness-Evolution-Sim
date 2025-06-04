from enum import Enum
import logging
import random

logger = logging.getLogger(__name__)


class ActionType(Enum):
    MOVE = "move"
    GATHER = "gather"
    COOPERATE = "cooperate"


class BaseAgent:
    """Agente base del sistema evolutivo."""

    def __init__(self, genotype=None, color="red"):
        self.genotype = genotype or []
        self.color = color
        # Cantidad de recursos recolectados por el agente
        self.inventory = 0

        # Métricas básicas utilizadas por las funciones de fitness
        self.resources_collected = 0
        self.shared_resources = 0
        self.alive = True
        self.steps_survived = 0

    def act(self, observation):
        """Define la acción del agente dado un estado de observación.

        Esta implementación básica permite que el agente pueda
        participar en la simulación sin necesidad de subclasses.
        El comportamiento es muy simple: si hay un recurso en la
        posición actual intentará recolectarlo; de lo contrario se
        moverá en una dirección aleatoria.
        """
        if observation.get("resource_here"):
            # logger.info("GATHER action triggered at %s", observation.get("position"))
            return Action(ActionType.GATHER)

        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        return Action(ActionType.MOVE, direction=(dx, dy))


class Action:
    """Representa una acción ejecutada por un agente."""

    def __init__(self, action_type, **kwargs):
        self.type = ActionType(action_type)
        self.params = kwargs
