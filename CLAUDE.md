# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es

Refinamiento y clasificación de octrees 3D contra una malla de triángulos (`cortex.mdl`). Cada cubo se clasifica como INSIDE / OUTSIDE (se poda) / BORDER (se subdivide en 8). Salida: VTK Legacy para ParaView. La documentación del repo (README, INSTRUCTIVO.txt, comentarios) está en español; mantener ese idioma en mensajes, comentarios y docs.

## Comandos

No hay linter ni framework de tests; la validación es la comparación cruzada de resultados entre versiones. Se requiere `nvcc` (GPU) y `g++` C++17 + OpenMP (CPU). Los Makefiles están pensados para Linux/WSL, no para PowerShell nativo.

```bash
# CUDA (raíz). ARCH por defecto sm_89; cambiar si la GPU es otra (p. ej. ARCH=sm_86)
make
make run LEVEL=5 MODE=uniform QUIET=1 MDL=cortex.mdl   # MODE=adaptive|uniform, QUIET=0|1, LEVEL 1..20
./octree <level> <mdl> [--quiet|-q] [--uniform|-u] [--adaptive|-a]

# CPU
make -C cpu_single run LEVEL=5 MDL=../cortex.mdl        # binario: cpu_single/octree_cpu
make -C cpu_multi  run LEVEL=5 THREADS=8                # binario: cpu_multi/octree_cpu_mt

# Verificar que las tres versiones coinciden exactamente (compila y ejecuta ambos modos)
./verificar.sh [nivel]          # nivel 5 por defecto

# Analizar un VTK: distribución por nivel y cumplimiento de la regla 2:1 (exit 0 = ok, 2 = violaciones)
make analyze archive=output/octree_pruned_level_5.vtk

# Limpiar todo
make clean && make -C cpu_single clean && make -C cpu_multi clean
```

`experimental/` usa además `SPLIT` (nivel de descomposición, default 1), `PASSES` y `VTK`. `experimental2/` tiene `make test` (compila y corre `tests/merge_test.cu`).

## Arquitectura

**Tres implementaciones independientes del mismo algoritmo** (`src/` CUDA, `cpu_single/src/`, `cpu_multi/src/`). No comparten código: cada una tiene su propio `main`, `octree`, `mesh_classify`, `stats` y `vtk_writer`. Un cambio de comportamiento o corrección del algoritmo normalmente debe replicarse en las tres, y `./verificar.sh` debe seguir reportando "las tres versiones coinciden" (el conteo `Dentro:`/`Borde:` final es la comparación). `sizeof(Cube)` difiere a propósito (48 B en CUDA, 24 B en CPU).

**Bucle principal** (`src/main.cu`): para cada etapa → refinar (`launchRefinement` uniforme o `launchAdaptiveRefinement`) → clasificar contra la malla (`launchClassificationMesh`) → podar OUTSIDE (`launchPrune`) → registrar tiempo/VRAM en `LevelStat`. Malla y cubos permanecen en VRAM; solo vuelven al host contadores durante las etapas y el arreglo final para exportar el VTK. No introducir copias device→host intermedias.

**Representación del octree en GPU**: arreglo plano de `struct Cube` (`src/octree.cuh`), sin punteros. `grid_index` (coordenadas enteras en el nivel) y `path_code` (clave Morton/base-8, raíz = 1, 3 bits por nivel, cabe hasta nivel 20) existen para localizar vecinos en device mediante búsqueda binaria sobre claves ordenadas, sin traer el árbol a RAM.

**Refinamiento adaptativo con balance 2:1**: solo se subdividen BORDER, más los vecinos INSIDE más gruesos (por cara, arista o vértice) necesarios para que ninguna pareja de hojas vecinas difiera en más de un nivel. `LEVEL` es profundidad máxima, no refinamiento total. El balance es una restricción **global**; esto importa en cualquier variante que procese el árbol por partes (ver `experimental/`).

**Estadísticas de memoria** (`stats.cu`): el pico "calculado" se deriva de tamaños de buffer (entrada + salida + scratch adaptativo, o salida + buffer de poda) y es el comparable entre niveles; el "medido" viene de `cudaMemGetInfo` y se redondea a bloques del driver. La línea base del contexto CUDA (~1 GB) se descuenta.

**Variantes experimentales** (copias del código de `src/` con un cambio de arquitectura; cada una tiene su propio README y Makefile):
- `experimental/`: descomposición por subárboles para reducir el pico de VRAM (nivel 11: 3,22 GB → 464 MB, ~2× más lento). Cada subárbol se procesa solo con un halo de "cáscaras" de vecinos adyacentes, que deben llevar la bandera `split` y guardarse por paso de refinamiento; luego `launchVerifyBalance` y repetición hasta 0 violaciones (típicamente 2 pasadas). Exporta VTK en streaming (`vtk_stream.cu`). El resultado debe coincidir hoja por hoja con el base. `mejoras.md` anota que `SPLIT>1` requiere más validación del balance.
- `experimental2/`: tras la poda, une hojas del mismo nivel que comparten cara completa y exporta la superficie como `POLYDATA`; no altera el refinamiento.
- `experimental3/` (binario `octree_dag`): guarda el octree en VRAM como DAG, compartiendo subárboles idénticos o iguales salvo espejo en X/Y/Z. Nodo de 32 B con 8 slots (tipo, máscara de espejo, índice); la geometría se deduce del camino y los vecinos se buscan con `dagLookup` descendiendo desde la raíz con la máscara acumulada por XOR. Con `SPLIT>0` (por defecto 1) el DAG se construye subárbol por subárbol contra un DAG global de índices estables, de modo que el arreglo completo de hojas nunca existe; en adaptive usa el halo de `experimental/` y repite pasadas hasta 0 violaciones. `SPLIT=0` es el camino monolítico y sirve de oráculo (`COMPARE=1`). Con `SPLIT>0` no se escribe VTK. Otras variables: `MIRROR`, `PASSES`, `PERFIL`, `ESFERA=R`, `REGEN`, `VTK`, `SIMILITUD` (solo mide similitud aproximada entre nodos y estima formatos delta, `src/similitud.cu`); `make test` corre `tests/dag_test.cu`. Su bucle de pasos corrige un `break` que sigue presente en `experimental/src/main.cu` y que rompe el balance con `SPLIT>1`.

Otros: `vtk_refinement_analyzer/` (verifica niveles y 2:1 sobre VTK binarios), `tools/oct_to_vtk.cpp`, `external/MixedOcTree` (submódulo de referencia, `.gitmodules`), `mejoras.md` (backlog de optimización GPU: BVH de triángulos, heredar INSIDE, costo del balance 2:1, reutilizar buffers), `resultados_pruebas.md` y `benchmark_logs/` (mediciones).

## Salidas y git

- VTK: `output/octree_{pruned|uniform}_level_N.vtk` (CUDA), `cpu_single/output/octree_cpu_single[_uniform]_level_N.vtk`, `cpu_multi/output/octree_cpu_multi[_uniform]_level_N.vtk`. Atributos: `state` (0 fuera, 1 dentro, 2 borde) y `level`.
- `.gitignore` excluye `output/`, `*.vtk`, `*.oct` (GitHub rechaza archivos >100 MB; las salidas llegan a ~6 GB). No añadirlos al repo.
- Desde Windows, `git` sobre `\\wsl.localhost\...` falla con "dubious ownership"; ejecutar git dentro de WSL o añadir el directorio a `safe.directory`.
