import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.patches import Rectangle


class Renderer:
    """Renderiza el mundo y los agentes utilizando matplotlib."""

    def __init__(self, record: bool = False, video_path: str = "sim.mp4"):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.record = record
        self.writer = None
        if self.record:
            if writers.is_available("ffmpeg"):
                self.writer = FFMpegWriter(fps=10)
                try:
                    self.writer.setup(self.fig, video_path)
                except FileNotFoundError:
                    logging.warning("No se encontró FFmpeg. Desactivando grabación.")
                    self.record = False
                    self.writer = None
            else:
                logging.warning("FFmpeg no está disponible. Desactivando grabación.")
                self.record = False


    def draw(
        self,
        world,
        generation: int | None = None,
        agent_index: int | None = None,
    ):
        """Dibuja el estado actual del mundo.

        Si se proporcionan ``generation`` o ``agent_index`` se muestran en el
        t\u00edtulo de la figura.
        """
        self.ax.clear()
        self.ax.set_xlim(0, world.width)
        self.ax.set_ylim(0, world.height)
        if world.grid:
            self.ax.set_xticks(range(world.width + 1))
            self.ax.set_yticks(range(world.height + 1))
            self.ax.grid(True, which="both")

        for zone in world.danger_zones:
            x, y = zone
            color = "red"
            if world.grid:
                rect = Rectangle((x, y), 1, 1, color=color, alpha=0.3)
            else:
                rect = Rectangle((x - 0.5, y - 0.5), 1, 1, color=color, alpha=0.3)
            self.ax.add_patch(rect)

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

        if getattr(world, "last_coop", None) is not None:
            x, y = world.last_coop
            if world.grid:
                x += 0.5
                y += 0.5
            self.ax.scatter(x, y, c="purple", s=150, marker="*")
        titulo = ["Simulación"]
        if generation is not None:
            titulo.append(f"generación {generation}")
        if agent_index is not None:
            titulo.append(f"individuo {agent_index}")
        self.ax.set_title(" - ".join(titulo))

        plt.draw()
        plt.pause(0.001)
        if self.record and self.writer:
            self.writer.grab_frame()

    def close(self):
        """Finaliza la grabación y cierra el writer."""
        if self.writer:
            self.writer.finish()
            self.writer = None
