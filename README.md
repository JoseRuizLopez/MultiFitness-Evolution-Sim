# MultiFitness-Evolution-Sim
Desarrollar un entorno donde múltiples agentes evolucionan mediante algoritmos genéticos o meméticos, con múltiples funciones objetivo (recompensas) en conflicto o balance: por ejemplo, velocidad, cooperación, diversidad genética, etc.

## Componentes principales del sistema
### Entorno simulado (tipo juego o mundo 2D)
Mapa 2D cuadriculado o continuo.

Recursos limitados y distribuibles (comida, energía...).

Obstáculos, zonas peligrosas.

Los recursos desaparecen al ser consumidos y se reponen automáticamente en una
posición libre del mapa.

Tiempo discretizado por "ticks".

### Agentes evolutivos
Cada agente tiene un "genotipo" (vector o estructura que codifica comportamiento).

El "fenotipo" es el comportamiento observable (movimiento, decisión...).

Comportamientos pueden codificarse como:

1. Red neuronal simple.
2. Máquina de estados finitos.
3. Secuencia de reglas con pesos.

En esta versión se define un genotipo como una lista de parámetros numéricos que
alimentan a una red neuronal mínima. Dicha red produce una de tres acciones
básicas:

* **Moverse**: desplaza al agente una casilla dentro del mundo.
* **Recolectar**: consume el recurso presente en la casilla actual y lo almacena
  en el inventario del agente.
* **Cooperar**: comparte un recurso del inventario con otro agente que se
  encuentre en la misma posición.

### Sistema evolutivo

#### Algoritmo Evolutivo Base: NSGA-II o MOEA/D
Para manejar múltiples objetivos sin necesidad de convertirlos en una única métrica.

- NSGA-II (Non-dominated Sorting Genetic Algorithm II): genera frentes de Pareto y mantiene diversidad mediante crowding distance.
- MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition): divide el problema multiobjetivo en subproblemas escalarizados.

Implementable desde cero o usando pymoo.
Se incluye una versión simplificada de NSGA-II para evolucionar la población.

### Métricas de recompensa posibles

- Eficiencia: velocidad para alcanzar metas.
- Diversidad genética: variación en la población.
- Cooperación: compartir recursos con otros.
- Crecimiento: cantidad de recursos recolectados.
- Supervivencia: número de agentes vivos tras N ticks.

Cada métrica puede tener un peso o función de penalización. Puedes experimentar con:

- Pareto dominance (frentes no dominados),
- Ponderaciones dinámicas.

Los pesos iniciales de cada métrica se encuentran en `multi_evo_sim/config.py`
mediante el diccionario `FITNESS_WEIGHTS`. Puedes modificarlos para priorizar
unas u otras métricas durante la evaluación.

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

## Ejecución rápida

1. Instala las dependencias principales:

   ```bash
   pip install -r requirements.txt
   ```

2. Ejecuta los tests:
    ```bash
    pytest -q
    ```

3. Ejecuta la simulación básica:

   ```bash
   python -m multi_evo_sim.main
   ```

4. Inicia el proceso de entrenamiento evolutivo:

   ```bash
   python -m multi_evo_sim.training
   ```

   Durante el entrenamiento se guardan tres archivos CSV: `fitness_log.csv`,
   `pareto_front.csv` e `inventory_log.csv`, donde este último almacena el
   inventario final de cada agente en cada generación.

El archivo `multi_evo_sim/config.py` contiene parámetros globales como el tamaño
del mundo, el número de agentes inicial y los pesos de cada métrica de fitness.
Modifica estos valores para experimentar con diferentes configuraciones.

## Entrenamiento de agentes

En esta versión se ha ajustado el diccionario `FITNESS_WEIGHTS` en
`multi_evo_sim/config.py` para dar más peso a las métricas de
`crecimiento` y `cooperacion`. Al reducir la influencia de
`eficiencia`, `diversidad_genetica` y `supervivencia`, el entrenamiento
refleja mejor las variaciones en la recolección y el intercambio de
recursos durante simulaciones cortas.
