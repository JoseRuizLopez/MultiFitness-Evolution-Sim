"""Funciones de fitness para distintos objetivos evolutivos."""

from ..config import FITNESS_WEIGHTS


def eficiencia(agent):
    """Calcula la eficiencia del agente.

    En este ejemplo se asume que genotipos más cortos representan
    comportamientos más eficientes.
    """
    return 1.0 / (1 + len(agent.genotype))


def diversidad_genetica(agent):
    """Evalúa la diversidad genética del agente basado en su genotipo."""
    if not agent.genotype:
        return 0.0
    return len(set(agent.genotype)) / len(agent.genotype)


def cooperacion(agent):
    """Cantidad de recursos que el agente ha compartido con otros."""
    return getattr(agent, "shared_resources", 0)


def crecimiento(agent):
    """Recursos totales recolectados por el agente."""
    return getattr(agent, "resources_collected", 0)


def supervivencia(agent):
    """Indica si el agente sigue con vida."""
    return 1.0 if getattr(agent, "alive", True) else 0.0


def fitness_combinado(agent):
    """Calcula el fitness total ponderado según la configuración."""
    scores = {
        "eficiencia": eficiencia(agent),
        "diversidad_genetica": diversidad_genetica(agent),
        "cooperacion": cooperacion(agent),
        "crecimiento": crecimiento(agent),
        "supervivencia": supervivencia(agent),
    }

    total = 0.0
    for nombre, valor in scores.items():
        peso = FITNESS_WEIGHTS.get(nombre, 1.0)
        total += valor * peso
    return [total]
