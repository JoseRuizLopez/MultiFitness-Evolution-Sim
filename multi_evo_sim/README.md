# MultiFitness Evolution Sim

Este directorio contiene el código fuente principal de la simulación de evolución con múltiples funciones objetivo.

Las funciones de fitness implementadas incluyen:

- `eficiencia`
- `diversidad_genetica`
- `cooperacion`
- `crecimiento`
- `supervivencia`

El peso de cada métrica puede configurarse en `config.py` mediante el diccionario
`FITNESS_WEIGHTS`.
## Entorno 2D

La clase `World` permite crear mapas en modo cuadriculado o en modo continuo. Es
posible configurar obstáculos, zonas peligrosas y recursos que se regeneran de
forma automática cuando son consumidos.

### Ejecutar la simulación

Desde la raíz del repositorio ejecuta:

```bash
python -m multi_evo_sim.main
```

Antes de lanzar la simulación puedes modificar `config.py` para ajustar el
tamaño del mundo, la población inicial o los pesos de las funciones de fitness.
