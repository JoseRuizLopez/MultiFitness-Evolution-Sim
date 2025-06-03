from typing import List, Tuple

from ..agents.base_agent import ActionType


class World:
    """Entorno 2D simple donde viven los agentes."""

    def __init__(self, width: int, height: int, resources=None):
        self.width = width
        self.height = height
        self.resources = resources or []
        # Cada elemento es una tupla (agente, (x, y))
        self.agents: List[Tuple[object, Tuple[int, int]]] = []

    def add_agent(self, agent, position):
        self.agents.append((agent, position))

    def step(self):
        """Avanza un tick en el mundo."""
        for idx, (agent, position) in enumerate(self.agents):
            observation = self.observe(agent, position)
            action = agent.act(observation)
            new_position = self.apply_action(agent, position, action)
            self.agents[idx] = (agent, new_position)

    def observe(self, agent, position):
        """Devuelve información sencilla del entorno para el agente."""
        resource_here = any(
            r.position == position and not r.consumed for r in self.resources
        )
        return {"resource_here": int(resource_here), "inventory": agent.inventory}

    def apply_action(self, agent, position, action):
        """Actualiza el mundo según la acción dada."""
        x, y = position

        if action.type == ActionType.MOVE:
            dx, dy = action.params.get("direction", (0, 0))
            nx = max(0, min(self.width - 1, x + dx))
            ny = max(0, min(self.height - 1, y + dy))
            return (nx, ny)

        if action.type == ActionType.GATHER:
            for res in self.resources:
                if res.position == position and not res.consumed:
                    agent.inventory += res.consume()
                    break
            return position

        if action.type == ActionType.COOPERATE:
            for other, other_pos in self.agents:
                if other is not agent and other_pos == position and agent.inventory > 0:
                    other.inventory += 1
                    agent.inventory -= 1
            return position

        return position
