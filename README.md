# MultiFitness-Evolution-Sim
Desarrollar un entorno donde múltiples agentes evolucionan mediante algoritmos genéticos o meméticos, con múltiples funciones objetivo (recompensas) en conflicto o balance: por ejemplo, velocidad, cooperación, diversidad genética, etc.

## Componentes principales del sistema
### Entorno simulado (tipo juego o mundo 2D)
Mapa 2D cuadriculado o continuo.

Recursos limitados y distribuibles (comida, energía...).

Obstáculos, zonas peligrosas.

Tiempo discretizado por "ticks".

### Agentes evolutivos
Cada agente tiene un "genotipo" (vector o estructura que codifica comportamiento).

El "fenotipo" es el comportamiento observable (movimiento, decisión...).

Comportamientos pueden codificarse como:

1. Red neuronal simple.
2. Máquina de estados finitos.
3. Secuencia de reglas con pesos.

### Sistema evolutivo

#### Algoritmo Evolutivo Base: NSGA-II o MOEA/D
Para manejar múltiples objetivos sin necesidad de convertirlos en una única métrica.

- NSGA-II (Non-dominated Sorting Genetic Algorithm II): genera frentes de Pareto y mantiene diversidad mediante crowding distance.
- MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition): divide el problema multiobjetivo en subproblemas escalarizados.

Implementable desde cero o usando pymoo.

### Métricas de recompensa posibles

- Eficiencia: velocidad para alcanzar metas.
- Diversidad genética: variación en la población.
- Cooperación: compartir recursos con otros.
- Crecimiento: cantidad de recursos recolectados.
- Supervivencia: número de agentes vivos tras N ticks.

Cada métrica puede tener un peso o función de penalización. Puedes experimentar con:

- Pareto dominance (frentes no dominados),
- Ponderaciones dinámicas.

### Tecnologías sugeridas
Entorno visual          ->  matplotlib (modo visualización)
Lógica de simulación    ->  Python puro (clases de mundo y agentes)
Evolución multiobjetivo ->	DEAP, pymoo, o implementación propia
Registro y análisis     ->	pandas, matplotlib, seaborn

### Estructura de carpetas sugerida
multi_evo_sim/

|

├── agents/

│   └── base_agent.py

│   └── neural_agent.py

│

├── env/

│   └── world.py

│   └── resource.py

│

├── evolution/

│   └── genetic_algorithm.py

│   └── fitness_functions.py

│

├── visualization/

│   └── render.py

│   └── logger.py

│

├── main.py

├── config.py

└── README.md

### Qué se puede investigar / demostrar
- Cómo cambia la población cuando priorizas diversidad vs rendimiento.
- Efecto de la cooperación vs comportamiento egoísta.
- Cuánto influye la búsqueda local en el resultado final.
- Visualización de frentes de Pareto en recompensas.
- Generación automática de estrategias sorprendentes (emergentes).
