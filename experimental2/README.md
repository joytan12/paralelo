# experimental2 — unión final de hojas CUDA

Esta variante conserva el refinamiento CUDA del proyecto base y añade una
etapa final en GPU. Las hojas supervivientes se agrupan cuando cumplen las dos
condiciones siguientes:

- tienen exactamente el mismo `level`;
- comparten una cara completa.

Los contactos solo por arista o vértice no forman una región. Las hojas de
distinto nivel tampoco se unen, aunque se toquen. Cada grupo forma una región
con forma libre compuesta por cubos: se eliminan sus caras internas y se
exportan sus caras externas como una superficie VTK `POLYDATA`.

El octree no se altera durante el refinamiento. La unión ocurre una vez que
termina la poda, por lo que el balance 2:1 y las estadísticas por nivel siguen
describiendo las hojas originales.

## Uso

Desde esta carpeta:

```bash
make
make test
make run LEVEL=5 QUIET=1
make run LEVEL=5 MODE=uniform QUIET=1
```

El modelo predeterminado es `../cortex.mdl`. El resultado se guarda como:

```text
output/octree_merged_adaptive_level_N.vtk
output/octree_merged_uniform_level_N.vtk
```

En ParaView se abre como una sola malla de superficie. El campo `region_id`
permite colorear cada región; `level`, `state` y `direction` describen la hoja
que emitió cada cara. Usa la representación `Surface` para ver la figura sin
las líneas de la teselación exterior.

Las caras entre cubos de la misma región se eliminan en GPU. En una frontera
entre niveles distintos se conserva una cara para cada región porque esas
hojas no pueden compartir región por diseño. La etapa final necesita memoria
para las caras visibles, por lo que su consumo depende de la superficie y no
reduce el pico de VRAM que ya requiere el refinamiento base.
