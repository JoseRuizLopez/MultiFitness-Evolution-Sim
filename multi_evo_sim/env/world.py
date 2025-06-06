from typing import List, Tuple
import random

from ..agents.base_agent import ActionType
from .resource import Resource


class World:
    """Entorno 2D donde viven los agentes.

    El mundo puede funcionar en modo cuadriculado (discreto) o continuo. En modo
    cuadriculado las posiciones se representan como tuplas ``(x, y)`` con valores
    enteros. En modo continuo se permiten ``float``.
    """

    def __init__(self, width, height, grid=True, resources=None, obstacles=None,
                 danger_zones=None, resource_regen=True):
        self.width = width
        self.height = height
        self.grid = grid

        # Colecciones de entidades del mundo
        self.resources = resources or []
        self.agents: List[Tuple[object, Tuple[int, int]]] = []
        self.obstacles = obstacles or []
        self.danger_zones = danger_zones or []

        self.resource_regen = resource_regen

    def add_agent(self, agent, position):
        """Registra un agente dentro del mundo en la posición indicada."""
        self.agents.append([agent, position])

    # ------------------------------------------------------------------
    # Gestión de elementos estáticos
    # ------------------------------------------------------------------
    def add_obstacle(self, obstacle):
        """Añade un obstáculo al mapa.

        Un obstáculo es una posición (modo cuadriculado) o una tupla
        ``(x1, y1, x2, y2)`` en modo continuo.
        """
        self.obstacles.append(obstacle)

    def add_danger_zone(self, zone):
        """Añade una zona peligrosa al mapa."""
        self.danger_zones.append(zone)

    def add_resource(self, resource):
        """Inserta un recurso ya creado en el mundo."""
        self.resources.append(resource)

    def spawn_resource(self, value=1):
        """Crea un recurso en una posición libre aleatoria."""
        pos = self._random_position()
        self.resources.append(Resource(pos, value))

    def spawn_danger_zone(self):
        """Genera una zona peligrosa en una posición libre."""
        pos = self._random_position()
        if self.grid:
            self.danger_zones.append(pos)
        else:
            x, y = pos
            x = min(x, self.width - 1)
            y = min(y, self.height - 1)
            self.danger_zones.append((x, y, x + 1, y + 1))

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _random_position(self):
        """Devuelve una posición válida aleatoria."""
        while True:
            x = random.uniform(0, self.width) if not self.grid else random.randrange(self.width)
            y = random.uniform(0, self.height) if not self.grid else random.randrange(self.height)
            pos = (x, y)
            if not self.is_obstacle(pos) and not self.is_danger(pos):
                return pos

    def is_obstacle(self, position):
        """Comprueba si una posición está ocupada por un obstáculo."""
        if self.grid:
            return position in self.obstacles
        for x1, y1, x2, y2 in self.obstacles:
            if x1 <= position[0] <= x2 and y1 <= position[1] <= y2:
                return True
        return False

    def is_danger(self, position):
        """Comprueba si la posición pertenece a una zona peligrosa."""
        if self.grid:
            return position in self.danger_zones
        for x1, y1, x2, y2 in self.danger_zones:
            if x1 <= position[0] <= x2 and y1 <= position[1] <= y2:
                return True
        return False

    def step(self):
        """Avanza un tick en el mundo y actualiza m\u00e9tricas b\u00e1sicas."""
        for idx, (agent, position) in enumerate(self.agents):
            if not getattr(agent, "alive", True):
                continue
            observation = self.observe(agent, position)
            action = agent.act(observation)
            new_position = self.apply_action(agent, position, action)
            agent.steps_survived = getattr(agent, "steps_survived", 0) + 1
            if self.is_danger(new_position):
                agent.alive = False
            self.agents[idx] = (agent, new_position)

        if self.resource_regen:
            for res in self.resources:
                if res.consumed:
                    res.position = self._random_position()
                    res.consumed = False

    def observe(self, agent, position):
        """Devuelve información del entorno para el agente."""
        resource_here = any(
            r.position == position and not r.consumed for r in self.resources
        )
        other_agents = [pos for a, pos in self.agents if a is not agent]
        return {
            "position": position,
            "resource_here": int(resource_here),
            "inventory": agent.inventory,
            "resources": [r.position for r in self.resources if not r.consumed],
            "danger": self.is_danger(position),
            "agents": other_agents,
            "obstacles": self.obstacles,
        }

    def apply_action(self, agent, position, action):
        """Actualiza el mundo según la acción dada."""
        x, y = position

        if isinstance(action, (list, tuple)) and len(action) == 2:
            dx, dy = action
            nx = x + dx
            ny = y + dy
            new_pos = (nx, ny)
            if 0 <= nx < self.width and 0 <= ny < self.height and not self.is_obstacle(new_pos):
                return new_pos
            return position

        if action.type == ActionType.MOVE:
            dx, dy = action.params.get("direction", (0, 0))
            nx = max(0, min(self.width - 1, x + dx))
            ny = max(0, min(self.height - 1, y + dy))
            return (nx, ny)

        if action.type == ActionType.GATHER:
            for res in self.resources:
                if res.position == position and not res.consumed:
                    value = res.consume()
                    agent.inventory += value
                    agent.resources_collected = getattr(agent, "resources_collected", 0) + value
                    break
            return position

        if action.type == ActionType.COOPERATE:
            for other, other_pos in self.agents:
                if other is not agent and other_pos == position and agent.inventory > 0:
                    other.inventory += 1
                    agent.inventory -= 1
                    agent.shared_resources = getattr(agent, "shared_resources", 0) + 1
            return position

        return position
