WORLD_WIDTH = 10
WORLD_HEIGHT = 10
POPULATION_SIZE = 10

# Algoritmo evolutivo por defecto. Si se establece en ``True`` se empleará
# ``MemeticNSGAII`` en ``training.py``. De lo contrario se usará
# ``GeneticAlgorithm``.
USE_MEMETIC_ALGORITHM = True

# Pesos de referencia para las métricas de fitness.
# NOTA: fitness_combinado() ya no usa estos pesos — devuelve los 5 objetivos
# por separado para que NSGA-II construya frentes de Pareto reales.
# Se mantiene aquí como referencia para experimentos con suma ponderada manual.
FITNESS_WEIGHTS = {
    "eficiencia": 0.5,
    "diversidad_genetica": 0.5,
    "cooperacion": 1.0,
    "crecimiento": 3.0,
    "supervivencia": 0.5,
}
