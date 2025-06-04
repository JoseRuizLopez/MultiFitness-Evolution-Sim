import logging
from typing import Iterable, List

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def log(message):
    logger.info(message)


class ExperimentLogger:
    """Registra estadísticas de fitness, inventarios y frentes de Pareto."""

    def __init__(
        self,
        fitness_file: str = "fitness_log.csv",
        pareto_file: str = "pareto_front.csv",
        inventory_file: str = "inventory_log.csv",
    ):
        self.fitness_file = fitness_file
        self.pareto_file = pareto_file
        self.inventory_file = inventory_file
        self._fitness_rows: List[dict] = []
        self._pareto_rows: List[dict] = []
        self._inventory_rows: List[dict] = []

    def log_fitness(self, iteration: int, fitness_values: Iterable):
        for idx, fit in enumerate(fitness_values):
            if not isinstance(fit, (list, tuple)):
                fit = [fit]
            row = {"iteration": iteration, "individual": idx}
            for j, val in enumerate(fit):
                row[f"obj_{j}"] = val
            self._fitness_rows.append(row)

    def log_pareto_front(self, iteration: int, front_indices: Iterable[int], fitness_values: Iterable):
        for idx in front_indices:
            fit = fitness_values[idx]
            if not isinstance(fit, (list, tuple)):
                fit = [fit]
            row = {"iteration": iteration, "individual": idx}
            for j, val in enumerate(fit):
                row[f"obj_{j}"] = val
            self._pareto_rows.append(row)

    def log_inventory(self, iteration: int, inventories: Iterable):
        for idx, inv in enumerate(inventories):
            row = {"iteration": iteration, "individual": idx, "inventory": inv}
            self._inventory_rows.append(row)

    def save(self):
        if self._fitness_rows:
            pd.DataFrame(self._fitness_rows).to_csv(self.fitness_file, index=False)
        if self._pareto_rows:
            pd.DataFrame(self._pareto_rows).to_csv(self.pareto_file, index=False)
        if self._inventory_rows:
            pd.DataFrame(self._inventory_rows).to_csv(self.inventory_file, index=False)
