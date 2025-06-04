WORLD_WIDTH = 10
WORLD_HEIGHT = 10
POPULATION_SIZE = 10

# Pesos para las distintas métricas de fitness
FITNESS_WEIGHTS = {
    # Las métricas que tienden a permanecer constantes en simulaciones
    # cortas reciben un peso menor para evitar que dominen el fitness
    # total. Se da mayor importancia a la cooperación y al crecimiento
    # para observar variaciones significativas.
    "eficiencia": 0.5,
    "diversidad_genetica": 0.5,
    "cooperacion": 1.0,
    "crecimiento": 3.0,
    "supervivencia": 0.5,
}
