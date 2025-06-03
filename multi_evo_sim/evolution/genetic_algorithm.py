class GeneticAlgorithm:
    """Implementación simplificada de un algoritmo genético multiobjetivo."""

    def __init__(self, population, fitness_fn):
        self.population = population
        self.fitness_fn = fitness_fn

    def step(self):
        """Realiza una iteración del algoritmo genético."""
        # Evaluar fitness
        fitness_scores = [self.fitness_fn(ind) for ind in self.population]
        # Selección, cruce y mutación aquí (simplificado)
        return fitness_scores
