
def example_fitness(agent):
    """Calcula dos objetivos simples basados en el genotipo."""
    if not agent.genotype:
        return [0.0, 0.0]
    obj1 = sum(agent.genotype)  # Maximizar la suma de genes
    # Maximizar la cercanía al valor 0.5 para cada gen
    obj2 = -sum((g - 0.5) ** 2 for g in agent.genotype) / len(agent.genotype)
    return [obj1, obj2]
