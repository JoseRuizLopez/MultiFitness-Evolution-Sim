class World:
    """Entorno 2D simple donde viven los agentes."""

    def __init__(self, width, height, resources=None):
        self.width = width
        self.height = height
        self.resources = resources or []
        self.agents = []

    def add_agent(self, agent, position):
        self.agents.append((agent, position))

    def step(self):
        """Avanza un tick en el mundo."""
        for agent, position in self.agents:
            observation = self.observe(agent, position)
            action = agent.act(observation)
            self.apply_action(agent, position, action)

    def observe(self, agent, position):
        """Observación simplificada para el ejemplo."""
        return {}

    def apply_action(self, agent, position, action):
        """Placeholder para procesar la acción del agente."""
        pass
