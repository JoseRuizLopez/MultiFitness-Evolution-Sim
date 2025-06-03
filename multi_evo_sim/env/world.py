import random
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
        self.obstacles = obstacles or []
        self.danger_zones = danger_zones or []

        self.resource_regen = resource_regen
        self.agents = []

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

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _random_position(self):
        """Devuelve una posición válida aleatoria."""
        while True:
            x = random.uniform(0, self.width) if not self.grid else random.randrange(self.width)
            y = random.uniform(0, self.height) if not self.grid else random.randrange(self.height)
            pos = (x, y)
            if not self.is_obstacle(pos):
                return pos

    def is_obstacle(self, position):
        """Comprueba si una posición está ocupada por un obstáculo."""
        if self.grid:
            return position in self.obstacles
        # en modo continuo obstáculos definidos como rectángulos (x1, y1, x2, y2)
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
        """Avanza un tick en el mundo."""
        for item in self.agents:
            agent, position = item
            observation = self.observe(agent, position)
            action = agent.act(observation)
            self.apply_action(agent, item, action)

        # Regeneración de recursos consumidos
        if self.resource_regen:
            for res in self.resources:
                if res.consumed:
                    res.position = self._random_position()
                    res.consumed = False

    def observe(self, agent, position):
        """Genera una observación básica para el agente."""
        obs = {
            "position": position,
            "resources": [r.position for r in self.resources if not r.consumed],
            "danger": self.is_danger(position),
        }
        return obs

    def apply_action(self, agent, item, action):
        """Procesa la acción del agente sobre el mundo."""
        # Por simplicidad solo manejamos acciones de movimiento expresadas
        # como desplazamiento (dx, dy)
        if isinstance(action, (list, tuple)) and len(action) == 2:
            new_x = item[1][0] + action[0]
            new_y = item[1][1] + action[1]
            new_pos = (new_x, new_y)

            if 0 <= new_x < self.width and 0 <= new_y < self.height and not self.is_obstacle(new_pos):
                item[1] = new_pos
