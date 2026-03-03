"""Funciones de fitness para distintos objetivos evolutivos."""


def eficiencia(agent):
    """Recursos recolectados por tick de supervivencia.

    Mide cuán eficientemente el agente aprovecha el tiempo disponible.
    Anteriormente usaba la longitud del genotipo, que es constante para
    todos los agentes y no diferenciaba ningún comportamiento.
    """
    steps = max(getattr(agent, "steps_survived", 1), 1)
    collected = getattr(agent, "resources_collected", 0)
    return collected / steps


def diversidad_genetica(agent):
    """Evalúa la diversidad genética del agente basado en su genotipo.

    Nota: esta es una aproximación por agente. La diversidad real es una
    métrica de población, pero aquí se usa como proxy individual para
    mantener la compatibilidad con la evaluación por agente.
    """
    if not agent.genotype:
        return 0.0
    return len(set(round(g, 2) for g in agent.genotype)) / len(agent.genotype)


def cooperacion(agent):
    """Cantidad de recursos que el agente ha compartido con otros."""
    return float(getattr(agent, "shared_resources", 0))


def crecimiento(agent):
    """Recursos totales recolectados por el agente."""
    return float(getattr(agent, "resources_collected", 0))


def supervivencia(agent):
    """Fracción de ticks que el agente sobrevivió respecto al total."""
    steps_survived = getattr(agent, "steps_survived", 0)
    # steps_survived se actualiza en world.step(); si el agente nunca
    # murió vale el total de pasos de la simulación.
    return 1.0 if getattr(agent, "alive", True) else steps_survived / max(steps_survived, 1)


def fitness_combinado(agent):
    """Devuelve los cinco objetivos por separado para NSGA-II.

    NSGA-II requiere múltiples objetivos para construir frentes de Pareto
    reales. Devolver un único valor escalar (suma ponderada) degrada el
    algoritmo a optimización mono-objetivo, desperdiciando la diversidad
    que aporta el crowding distance.

    Returns
    -------
    list[float]
        [eficiencia, diversidad_genetica, cooperacion, crecimiento, supervivencia]
    """
    return [
        eficiencia(agent),
        diversidad_genetica(agent),
        cooperacion(agent),
        crecimiento(agent),
        supervivencia(agent),
    ]
