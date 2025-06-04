import matplotlib.pyplot as plt


class Renderer:
    """Renderiza el mundo y los agentes utilizando matplotlib."""

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()

    def draw(self, world):
        """Dibuja el estado actual del mundo."""
        self.ax.clear()
        self.ax.set_xlim(0, world.width)
        self.ax.set_ylim(0, world.height)
        if world.grid:
            self.ax.set_xticks(range(world.width + 1))
            self.ax.set_yticks(range(world.height + 1))
            self.ax.grid(True, which="both")

        for res in world.resources:
            if not res.consumed:
                x, y = res.position
                if world.grid:
                    x += 0.5
                    y += 0.5
                self.ax.scatter(x, y, c="green", s=100, marker="s")

        for agent, pos in world.agents:
            x, y = pos
            if world.grid:
                x += 0.5
                y += 0.5
            color = getattr(agent, "color", "red")
            self.ax.scatter(x, y, c=color, s=100, marker="o")

        self.ax.set_title("Simulación")
        plt.draw()
        plt.pause(0.001)
