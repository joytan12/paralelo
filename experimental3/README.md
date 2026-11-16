# experimental3 — el octree final como DAG con espejos

El octree final del proyecto base es un arreglo plano de hojas `Cube` (48 B cada
una). Aquí se guarda como un **DAG** (grafo acíclico dirigido) en VRAM: los
subárboles idénticos, o idénticos salvo una reflexión en X, Y o Z, se almacenan
**una sola vez**. Una esfera centrada es el caso ideal: sus 8 octantes son
espejos entre sí y el nivel 1 se reduce a **un único nodo**.

Se construye además **sin que exista nunca el arreglo completo de hojas**
(descomposición por subárboles, también en adaptativo con halo de cáscaras), de
modo que el pico de VRAM deja de depender del tamaño del octree: cortex adaptativo
llega al nivel 13 (462 M hojas) con 1,97 GB, y uniforme al nivel 11 (2.331 M
hojas) con 1,52 GB, donde el base necesitaría decenas de GB.

Todo el cómputo es CUDA (kernels y thrust sobre device). Al host solo llegan
contadores enteros; los nodos y las hojas no se copian a RAM. La única excepción
es el camino monolítico (`--split-level 0`), que conserva la copia de las hojas
al host para escribir el VTK, igual que el base.

---

## Uso

Desde esta carpeta (Linux/WSL, `nvcc`; `ARCH=sm_89` por defecto):

```bash
make                         # compila ./octree_dag
make test                    # pruebas (tests/dag_test.cu)
make run LEVEL=9 QUIET=1     # adaptativo, DAG por subárboles (SPLIT=1)
make run LEVEL=9 MODE=uniform SPLIT=2 QUIET=1
make run LEVEL=11 PERFIL=1 QUIET=1       # perfil de tiempos por fase y pasada
make run LEVEL=9 ESFERA=0.7 QUIET=1      # esfera de radio 0,7 en lugar de la malla
make run LEVEL=9 SPLIT=0                 # camino monolítico (exporta VTK)
./octree_dag <nivel> [<mdl>] [opciones]
```

| Opción | Makefile | Efecto |
|---|---|---|
| `-q`, `--quiet` | `QUIET=1` | omite las líneas por etapa/subárbol |
| `-u` / `-a` | `MODE=uniform\|adaptive` | refinamiento uniforme o adaptativo con balance 2:1 (por defecto) |
| `--split-level D` | `SPLIT=D` | DAG por subárboles con descomposición en el nivel `D`; **por defecto 1**; `0` = camino monolítico |
| `--no-mirror` | `MIRROR=0` | sin simetrías de espejo (solo subárboles idénticos) |
| `--passes N` | `PASSES=N` | adaptativo: pasadas máximas de cierre del balance (8) |
| `--comparar` | `COMPARE=1` | con `D>0`: compara contra el DAG monolítico (uniforme ≤ 9, adaptativo ≤ 11) |
| `--no-regen` | `REGEN=0` | uniforme con `D>0`: omite la regeneración de cada bloque |
| `--perfil` | `PERFIL=1` | adaptativo con `D>0`: tiempo por fase y pasada |
| `--esfera R` | `ESFERA=R` | esfera de radio `R` centrada en el origen, raíz `half_size` 1 (`launchClassification(Sphere)`) en lugar de la malla |
| `--no-vtk` | `VTK=0` | no escribe el VTK (solo tiene efecto con `D=0`) |

> **VTK y `SPLIT`.** Con `SPLIT>0` (el valor por defecto) **no se escribe VTK**:
> las hojas completas nunca están en VRAM. `SPLIT=0` da el comportamiento del
> hito 1: arreglo completo de hojas, tabla de refinamiento, **dos** DAG (sin y
> con espejo) y VTK en `output/octree_{pruned|uniform}_level_N.vtk` (salvo
> `VTK=0`). El programa y `make help` lo repiten. Con `SPLIT>0` se construye un
> solo DAG: con espejo, o sin él con `--no-mirror`.

---

## Formato del nodo

Un nodo interno son 8 *slots* de 32 bits, uno por octante (`struct DagNode`,
32 B). El octante usa el orden de `refineKernel` / `path_code`:
`octante = bit_x | bit_y << 1 | bit_z << 2`.

```text
slot de 32 bits
  bits 31..30  tipo: 0 VACÍO (podado), 1 HOJA_INSIDE, 2 HOJA_BORDER, 3 NODO
  bits 29..27  máscara de espejo (bit 0 = X, bit 1 = Y, bit 2 = Z)
  bits 26..0   índice al arreglo de nodos únicos del nivel siguiente
```

La geometría (centro, tamaño, `grid_index`) **no se guarda**: se deduce del camino
desde la raíz. Los nodos únicos viven en un arreglo por nivel y la raíz es un slot.
El índice de 27 bits admite ~134 M nodos únicos por nivel (la construcción aborta
con un mensaje si no alcanza).

## Espejos y estabilizadores

`NODO(idx, m)` significa "el nodo `idx` visto a través de la reflexión `m`". Si `C`
es el nodo guardado, la vista es

```text
V[o] = reflejar_m( C[o ^ m] )
reflejar_m(NODO(idx2, m2)) = NODO(idx2, m2 ^ m);   hojas y vacíos no cambian
```

Las reflexiones forman un grupo con XOR, así que la máscara se acumula con XOR al
descender. `dagLookup(vista, x, y, z, nivel)` (`src/dag.cuh`) desciende desde la
raíz tomando el octante de las coordenadas enteras, aplica `o ^ acumulada` y, si el
slot es NODO, hace `acumulada ^= máscara_del_slot`. Devuelve el tipo y el nivel de
la hoja que cubre la celda (`DAG_NODO` si la celda está subdividida, `DAG_FUERA`
si cae fuera del dominio).

**Canonicalización.** De las 8 reflexiones de un nodo se elige la
lexicográficamente menor (sobre sus 8 `uint32`) y se guarda la máscara usada: si
`C = V_m(O)`, la referencia a `O` es `(idx(C), m)`. La deduplicación es
`sort + unique` con igualdad **exacta** de los 8 `uint32`; no se usa hash como
identidad.

**Estabilizadores.** Si un hijo es simétrico bajo una reflexión `s` (p. ej. un nodo
con 8 hojas iguales), `NODO(idx, m)` y `NODO(idx, m ^ s)` son el mismo subárbol. Sin
tratarlo, los padres de dos subárboles espejados no se reconocen como iguales (en la
esfera centrada el nivel 1 quedaba con 8 nodos únicos en vez de 1). Por cada nodo
único se guarda un byte con su **estabilizador** (las reflexiones que lo dejan
invariante) y toda máscara de un hijo se normaliza a la menor de su clase al armar
las vistas; en empates se elige la máscara menor, de modo que el mismo contenido
siempre produce la misma referencia.

---

## Construcción

### Monolítica (`--split-level 0`, `src/dag.cu` → `dagBuild`)

De abajo hacia arriba, nivel por nivel, desde el nivel máximo hasta 1: las
entradas del nivel `L` son las hojas de ese nivel más los nodos de nivel `L` ya
resueltos como referencia `(idx, m)`; se ordenan por `path_code`, los hermanos
(`path_code >> 3`) forman el padre, se canonicaliza, se deduplica y cada padre pasa
como entrada del nivel `L-1`. Parte del arreglo final de hojas, así que el pico de
VRAM sigue incluyéndolo. Se conserva como referencia independiente (oráculo) de las
variantes por subárboles.

### Incremental (`src/dag_global.inc`)

Un `DagGlobal` agrega subárboles de a uno. Por nivel guarda nodos con **índices
estables y solo-agregar** (un nodo ya guardado nunca cambia de índice, porque los
niveles superiores de otros subárboles lo referencian), su estabilizador y una
**permutación ordenada por contenido** para buscar. Insertar un lote: armar los
padres con los índices globales de sus hijos, canonicalizar con los estabilizadores
globales (nunca con índices locales que luego se remapeen: el orden lexicográfico
depende de ellos), `sort + unique`, búsqueda binaria contra la permutación,
agregar los que faltan y `thrust::merge` de la permutación. Los arreglos crecen
duplicando con copia device a device.

Los conteos (`Dentro`, `Borde`, hojas y nodos lógicos por nivel) se obtienen **sin
expandir**, propagando de arriba hacia abajo la multiplicidad de cada nodo único.

### Por subárboles, uniforme (`--uniform --split-level D`, hito 2a)

- **Fase A**: refinar/clasificar/podar hasta el nivel `D` (≤ 8^D raíces de subárbol).
- **Fase B**: por cada raíz, refinar solo ese subárbol hasta el nivel máximo,
  comprimirlo contra el `DagGlobal` y liberar sus hojas antes del siguiente. En
  uniforme no hay búsqueda de vecinos, así que los subárboles son independientes.
- **Fase C**: con las referencias de los subárboles se construyen los niveles `D..1`
  y la raíz.

### Por subárboles, adaptativo con halo y pasadas (`--adaptive --split-level D`, hito 2b)

Mecanismo portado de `experimental/` (copiado en `src/octree_halo.inc`; no se
enlaza ni se modifica `experimental/`). El balance 2:1 es global, así que:

- **Halo**: cada subárbol se refina sobre `[hojas propias | halo]`. El halo son las
  **cáscaras** (hojas que tocan la frontera exterior) de los subárboles
  **adyacentes en ese mismo paso** de refinamiento (lockstep), guardadas por paso y
  con la **bandera `split`** que su dueño calculó. La bandera es imprescindible: una
  hoja de frontera puede estar marcada por el balance interno de su subárbol, algo
  que no se puede reconstruir desde `state` y `level`.
- **Pasadas**: la primera pasada tiene el halo vacío y puede dejar violaciones en las
  fronteras. Se verifica la regla 2:1 sobre la unión de las cáscaras finales; con
  violaciones, la pasada **se descarta** (también su `DagGlobal`) y se repite con las
  cáscaras recién medidas; con 0, el DAG de esa pasada es el resultado. Cada pasada
  construye su propio `DagGlobal`. Converge en 2 pasadas en cortex.
- **Diferencia con `experimental/`**: el resultado de cada subárbol no se evacúa a
  disco, se comprime contra el DAG de la pasada y se liberan sus hojas. La poda es
  in-place (`launchPruneInPlace`).
- Las cáscaras de todos los pasos viven en VRAM y están contadas en el pico
  calculado (de 3,7 MB en el nivel 9 a 375 MB como máximo en el nivel 13).

**Corrección respecto al bucle de `experimental/`.** Un subárbol que no se divide en
un paso no debe dejar de participar: en un paso posterior el halo de sus vecinos
puede marcarlo (p. ej. una raíz INSIDE gruesa junto a celdas que se afinan después).
Cortar el bucle (`break`) dejaba con `SPLIT>1` un punto fijo equivocado (violaciones
constantes, hojas INSIDE de menos). Aquí cada subárbol recorre todos los pasos.

---

## Validaciones (todas en GPU)

- **Descompresión** (camino monolítico): se expande el DAG a hojas
  `(path_code, nivel, estado)`, se ordena y se compara elemento a elemento con el
  arreglo original. **Consultas**: `dagLookup(grid_index, level)` de cada hoja
  original debe devolver su estado y nivel.
- **Balance 2:1 global por bloques** (adaptativo): se expande el DAG subárbol por
  subárbol, cada hoja consulta sus 26 vecinos con `dagLookup` sobre el DAG
  completo y el bloque se libera. Es independiente del chequeo de cáscaras y es la
  validación principal donde no hay monolítico (niveles 12 y 13).
- **Por bloques con regeneración** (uniforme): cada bloque expandido se compara con el
  subárbol regenerado desde la malla. No aplica en adaptativo (un subárbol no se
  regenera solo sin su halo).
- **`--comparar`**: oráculo monolítico (uniforme: `pasoUniforme` + `dagBuild`;
  adaptativo: `launchAdaptiveRefinement` sin halo + `dagBuild`): mismas hojas
  expandidas, mismos nodos únicos por nivel y mismos conteos.
- Conteos `Dentro:`/`Borde:` iguales a `../octree` donde el base cabe.
- Contadores por multiplicidad iguales a las hojas recorridas por bloques.

## Pruebas (`make test`)

Casos armados a mano (8 hojas iguales, nodos espejados en X/Y/Z/XYZ, hijo con
simetría propia, hermanos que difieren en una rama, balance correcto y violado como
control negativo, raíz hoja, árbol vacío); esfera centrada (1 nodo único en el nivel
1 con espejo, 8 sin espejo) en uniforme y adaptativo; **fusión de subárboles** (idénticos,
espejados, nodos nuevos en un nivel poblado) y 12 casos pseudoaleatorios con
`split` 1..3 con capacidad inicial mínima (fuerza el crecimiento de los arreglos
globales); esfera por subárboles con `SPLIT` 1 y 2; esfera adaptativa grande con
raíces INSIDE (regresión del `break`), esferas descentradas que **obligan a una
segunda pasada**, y los casos de la fase A (ver limitaciones). Se comprobó con
mutaciones manuales (búsqueda binaria invertida, comparador del merge invertido,
`break` restaurado) que las pruebas fallan.

---

## Resultados

Máquina: RTX 4050 Laptop (6 GB), WSL2, `sm_89`. "Pico calculado" = malla +
cáscaras + buffers del refinamiento + hojas del subárbol + DAG global (sin el
temporal interno de thrust); "pico medido" = `cudaMemGetInfo`. Ningún caso
desbordó VRAM a memoria compartida (máximo medido: 3,34 GB).

### cortex.mdl

**Camino monolítico** (`SPLIT=0`; DAG = nodos únicos × 32 B):

| Modo | Nivel | Hojas | Base (×48 B) | Árbol sin compartir (nodos) | DAG sin espejo (únicos) | DAG con espejo (únicos) |
|---|---:|---:|---:|---:|---:|---:|
| adaptativo | 5 | 5.501 | 257,86 KB | 36,25 KB (1.160) | 14,19 KB (454) | 8,94 KB (286) |
| adaptativo | 7 | 105.634 | 4,84 MB | 640,56 KB (20.498) | 131,50 KB (4.208) | 107,22 KB (3.431) |
| adaptativo | 9 | 1.774.809 | 81,24 MB | 10,39 MB (340.328) | 1,39 MB (45.551) | 938,31 KB (30.026) |
| uniforme | 5 | 10.646 | 499,03 KB | 59,22 KB (1.895) | 14,22 KB (455) | 8,97 KB (287) |
| uniforme | 7 | 596.068 | 27,29 MB | 2,76 MB (90.560) | 128,47 KB (4.111) | 105,75 KB (3.384) |
| uniforme | 9 | 36.775.635 | 1,64 GB | 162,98 MB (5.340.446) | 1,36 MB (44.705) | 920,34 KB (29.451) |

**Uniforme nivel 9 — efecto de `SPLIT`** (tiempo y pico incluyen el refinamiento):

| SPLIT | Tiempo | Pico calc. / medido | Subárbol más grande | DAG con espejo |
|---:|---:|---:|---:|---:|
| 0 | 12,60 s | 3,33 GB / 3,34 GB | todo (36,8 M hojas) | 920,34 KB |
| 1 | 12,59 s | 539,5 MB / 544 MB | 5,8 M hojas (266 MB) | 920,34 KB |
| 2 | 13,67 s | 193,5 MB / 198 MB | 2,1 M hojas (96 MB) | 920,34 KB |

**Más profundo que el base**:

| Modo | Nivel | SPLIT | Hojas (multiplicidad) | Tiempo | Pico calc. / medido | DAG con espejo |
|---|---:|---:|---:|---:|---:|---:|
| uniforme | 10 | 2 | 292.390.073 | 98,6 s | 1,50 GB / 1,51 GB | 2,49 MB (81.734) |
| uniforme | 11 | 3 | 2.331.869.661 | 786,1 s | 1,51 GB / 1,52 GB | 7,63 MB (250.118) |
| adaptativo | 12 | 1 | 115.425.336 | 114,9 s | 1,25 GB / 1,38 GB | 24,07 MB (788.887) |
| adaptativo | 12 | 2 | 115.425.336 | 120,2 s | 537 MB / 612 MB | 24,07 MB (788.887) |
| adaptativo | 13 | 2 | 462.261.737 | 468,5 s | 1,78 GB / 1,97 GB | 69,51 MB (2.277.691) |

**Adaptativo por subárboles** (2 pasadas en todos los casos; violaciones de la
pasada 1 → 0):

| Nivel | SPLIT | Tiempo | Pico calc. / medido | Violaciones pasada 1 | Cáscaras máx. / final | DAG con espejo |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0 | 0,88 s | 204 MB / 212 MB | — | — | 938,31 KB (30.026) |
| 9 | 1 | 2,27 s | 27,0 MB / 36 MB | 21.222 | 7,2 MB / 3,7 MB | 938,31 KB |
| 9 | 2 | 5,21 s | 32,8 MB / 36 MB | 64.767 | 19,3 MB / 9,9 MB | 938,31 KB |
| 11 | 0 | 14,65 s | 3,22 GB / 3,23 GB | — | — | 7,77 MB (254.532) |
| 11 | 1 | 29,08 s | 336,7 MB / 378 MB | 106.011 | 32,3 MB / 16,5 MB | 7,77 MB |
| 11 | 2 | 32,70 s | 179,6 MB / 216 MB | 239.750 | 88,8 MB / 45,7 MB | 7,77 MB |
| 12 | 1 | 114,9 s | 1,25 GB / 1,38 GB | 219.297 | 66,4 MB / 33,9 MB | 24,07 MB (788.887) |
| 12 | 2 | 120,2 s | 537 MB / 612 MB | 478.139 | 183,7 MB / 94,6 MB | 24,07 MB |
| 13 | 2 | 468,5 s | 1,78 GB / 1,97 GB | 949.839 | 375,3 MB / 193,2 MB | 69,51 MB (2.277.691) |

Hojas del nivel 12: 115.425.336 (Dentro 57.572.261 / Borde 57.853.075), iguales a las
de `experimental/bench/BENCHMARK.md`; nivel 13: 462.261.737 (230.840.319 / 231.421.418).
Validaciones: 16 corridas de `--comparar` (niveles 5, 7, 9, 11 × `SPLIT` 1 y 2 × con y
sin espejo) idénticas al monolítico; balance global 0 en todas.

### Esfera centrada en el origen (`--esfera 0.7`, raíz `half_size` 1)

Es el caso ideal: **con espejos el nivel 1 queda en 1 nodo único** (8 sin espejo) y
el DAG es ~8× menor que sin espejo. En cortex (sin simetría) los espejos aportan solo
1,35× a 1,52×.

Adaptativo, nivel 9 (1.179.024 hojas, 1 pasada, `SPLIT=1`):

| Nivel | Hojas | Nodos lógicos | Únicos sin espejo | Únicos con espejo |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 1 | 1 |
| 1 | 0 | 8 | 8 | **1** |
| 2 | 0 | 32 | 32 | 4 |
| 3 | 0 | 160 | 153 | 20 |
| 4 | 136 | 928 | 733 | 94 |
| 5 | 2.048 | 4.024 | 2.499 | 317 |
| 6 | 11.104 | 15.776 | 8.779 | 1.114 |
| 7 | 50.696 | 55.808 | 18.957 | 2.459 |
| 8 | 216.848 | 151.400 | 204 | 36 |
| 9 | 898.192 | 0 | 0 | 0 |
| **Total** | **1.179.024** | **228.137** | **31.366** | **4.046** |

Adaptativo, nivel 11 (19.298.560 hojas, 1 pasada, `SPLIT=1`):

| Nivel | Hojas | Nodos lógicos | Únicos sin espejo | Únicos con espejo |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 1 | 1 |
| 1 | 0 | 8 | 8 | **1** |
| 2 | 0 | 32 | 32 | 4 |
| 3 | 0 | 160 | 153 | 20 |
| 4 | 136 | 928 | 715 | 94 |
| 5 | 1.928 | 4.144 | 2.731 | 346 |
| 6 | 10.424 | 17.416 | 10.391 | 1.305 |
| 7 | 50.080 | 69.544 | 37.141 | 4.678 |
| 8 | 214.816 | 263.320 | 121.357 | 15.346 |
| 9 | 889.192 | 904.360 | 78.529 | 9.934 |
| 10 | 3.599.296 | 2.421.200 | 204 | 36 |
| 11 | 14.532.688 | 0 | 0 | 0 |
| **Total** | **19.298.560** | **3.681.113** | **251.262** | **31.765** |

Uniforme, nivel 9 (24.407.824 hojas, `SPLIT=1`):

| Nivel | Nodos lógicos | Únicos sin espejo | Únicos con espejo |
|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 |
| 1 | 8 | 8 | **1** |
| 2 | 32 | 32 | 4 |
| 3 | 160 | 129 | 17 |
| 4 | 1.064 | 585 | 74 |
| 5 | 7.160 | 2.273 | 285 |
| 6 | 51.968 | 8.585 | 1.086 |
| 7 | 396.040 | 18.957 | 2.459 |
| 8 | 3.090.104 | 205 | 37 |
| **Total** | **3.546.537** | **30.775** | **3.964** |

Uniforme, nivel 11 (1.547.548.752 hojas, `SPLIT=3`):

| Nivel | Nodos lógicos | Únicos sin espejo | Únicos con espejo |
|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 |
| 1 | 8 | 8 | **1** |
| 2 | 32 | 32 | 4 |
| 3 | 160 | 129 | 17 |
| 4 | 1.064 | 585 | 74 |
| 5 | 7.160 | 2.385 | 299 |
| 6 | 51.968 | 9.281 | 1.161 |
| 7 | 396.040 | 35.383 | 4.441 |
| 8 | 3.090.104 | 119.691 | 15.128 |
| 9 | 24.407.824 | 78.529 | 9.934 |
| 10 | 194.048.208 | 205 | 37 |
| **Total** | **222.002.569** | **246.229** | **31.097** |

Tamaños (DAG = únicos × 32 B; los DAG no incluyen el slot raíz de 4 B):

| Caso | Hojas | Base (×48 B) | Árbol sin compartir | DAG sin espejo | DAG con espejo | Espejo aporta |
|---|---:|---:|---:|---:|---:|---:|
| adaptativo 9 | 1.179.024 | 53,97 MB | 6,96 MB | 980,19 KB | 126,44 KB | 7,75× |
| adaptativo 11 | 19.298.560 | 883,42 MB | 112,34 MB | 7,67 MB | 992,66 KB | 7,91× |
| uniforme 9 | 24.407.824 | 1,09 GB | 108,23 MB | 961,72 KB | 123,88 KB | 7,76× |
| uniforme 11 | 1.547.548.752 | 69,18 GB | 6,62 GB | 7,51 MB | 971,78 KB | 7,92× |
| cortex adaptativo 9 | 1.774.809 | 81,24 MB | 10,39 MB | 1,39 MB | 938,31 KB | 1,52× |
| cortex adaptativo 11 | 28.786.873 | 1,29 GB | 167,7 MB | 10,52 MB | 7,77 MB | 1,35× |

La esfera usa 8 de las 48 simetrías del cubo (solo reflexiones): el espejo reduce
~7,9× (el máximo con 8 reflexiones es 8×). Tiempo y pico con espejo: adaptativo 11,
0,97 s y 180/212 MB; uniforme 11 (`SPLIT=3`), 29,0 s y 1,50/1,51 GB (la clasificación
analítica es mucho más barata que la de la malla).

### Perfil de tiempos (cortex, adaptativo, `--perfil`, segundos)

La clasificación contra la malla domina (87–90 %). Nivel 11, `SPLIT=1`:

| Fase | Pasada 1 | Pasada 2 | Total | % |
|---|---:|---:|---:|---:|
| armar halo | 0,008 | 0,033 | 0,041 | 0,1 |
| refinar | 0,710 | 0,777 | 1,487 | 5,2 |
| cáscaras | 0,110 | 0,112 | 0,222 | 0,8 |
| clasificar | 12,748 | 12,799 | 25,547 | 89,2 |
| podar | 0,059 | 0,059 | 0,118 | 0,4 |
| insertar en DAG | 0,581 | 0,611 | 1,192 | 4,2 |
| verificar cáscaras | 0,008 | 0,009 | 0,017 | 0,1 |
| fase C | 0,000 | 0,002 | 0,002 | 0,0 |
| **suma** | 14,229 | 14,408 | 28,637 | 100 |

Nivel 12, `SPLIT=1`: clasificar 104,2 s de 115,1 s (90,5 %), insertar en DAG 3,6 s
(3,1 %), refinar 6,1 s (5,3 %). Con `SPLIT=2`: clasificar 87–80 %, insertar en DAG 5–9,5 %.
Medido en la misma sesión, `experimental/` (`SPLIT=1`, `--no-vtk`) tarda 27,8 s en el
nivel 11 y 110,9 s en el 12, contra 28,6 s y 115,1 s aquí: la diferencia (+3–4 %) es
la compresión contra el DAG.

### Similitud aproximada entre nodos (`--similitud`, hito 4, solo medición)

El DAG solo comparte coincidencias exactas (salvo espejo). `--similitud`
(`SIMILITUD=1`) mide, sin cambiar el formato, cuánto se ganaría guardando un nodo
"casi igual" a otro como diferencia (delta). Todo en GPU (`src/similitud.cu`).

Distancia entre dos nodos = slots distintos (0 a 8), mínimo sobre las 8 reflexiones;
dos referencias equivalentes por el estabilizador del hijo cuentan como iguales.

- **M1** distancia de cada nodo a su hermano más parecido (hijos `NODO` distintos de un
  mismo padre).
- **M2** nodos con otro nodo del mismo nivel a distancia 1, o a distancia 2.
- **M3** tamaño estimado de un formato delta (nodo 32 B, delta de 1 slot 8 B, de 2
  slots 16 B). Cota optimista: todo nodo con vecino a distancia 1 o 2 se vuelve delta
  (no alcanzable: la base de un delta debe ser un nodo completo). Cota conservadora:
  una sola posición de slot por nivel, 1 base y el resto deltas en cada grupo.
- **M4** similitud geométrica: cada nodo se expande a una rejilla de 8×8×8 celdas y se
  cuenta el porcentaje de celdas distintas respecto del hermano más parecido.
- **M5** tamaño con nodos de tamaño variable (cabecera de 2 B + 4 B por hijo `NODO`),
  sin deltas.

Adaptativo nivel 9, `--split-level 0`. Las mallas distintas de cortex son las de
`external/MixedOcTree/data/`. Los porcentajes son respecto del DAG actual con espejo.

| Malla | Hojas | Únicos con espejo | Hermano a d=1 | Nivel a d=1 | M5 | Delta conserv. | Delta optim. |
|---|---:|---:|---:|---:|---:|---:|---:|
| cortex | 1.774.809 | 30.026 | 2.008 (6,7 %) | 15.417 (51,3 %) | 76,9 % | 84,1 % | 55,2 % |
| femur | 347.439 | 6.584 | 760 (11,5 %) | 3.330 (50,6 %) | 71,1 % | 87,7 % | 52,6 % |
| fertilityRotated | 1.204.592 | 21.687 | 1.374 (6,3 %) | 11.604 (53,5 %) | 75,0 % | 84,8 % | 52,8 % |
| sein | 1.105.017 | 17.931 | 1.336 (7,5 %) | 9.292 (51,8 %) | 75,3 % | 84,5 % | 54,4 % |
| sc | 176.244 | 1.997 | 164 (8,2 %) | 1.138 (57,0 %) | 67,3 % | 87,6 % | 49,1 % |
| suture | 87.328 | 638 | 82 (12,9 %) | 328 (51,4 %) | 61,4 % | 89,3 % | 47,3 % |
| hex_sphe | 4.523.174 | 4.096 | 365 (8,9 %) | 2.103 (51,3 %) | 71,3 % | 89,4 % | 50,8 % |
| cyl50 | 711.264 | 160 | 9 (5,6 %) | 33 (20,6 %) | 58,8 % | 92,5 % | 58,6 % |
| esfera 0,7 | 1.179.024 | 4.046 | 331 (8,2 %) | 2.091 (51,7 %) | 70,6 % | 88,0 % | 50,4 % |

Lectura:

- **Entre hermanos hay poca similitud** en todas las mallas: solo 6 a 13 % de los nodos
  tiene un hermano a un slot de distancia.
- **En todo el nivel hay mucha más**: cerca de la mitad de los nodos tiene otro nodo del
  nivel a un slot de distancia. Se concentra dos niveles por encima de las hojas (en
  cortex, nivel 7: 11.924 de 12.885 nodos).
- **El ahorro de un formato delta queda entre 16 % y 45 %** del DAG en cortex (cotas
  conservadora y optimista). Un empaquetado de nodos de tamaño variable, sin búsqueda de
  parecidos, ya ahorra 23 %.
- **M4** (cortex): en los niveles 5 a 7 la mayoría de los nodos difiere de su hermano más
  parecido en más de 5 % de las celdas; la similitud geométrica entre hermanos tampoco es
  alta.
- La medición tarda menos de 0,3 s por malla en este nivel. Solo se midió hasta el nivel
  9 (uniforme, nivel 7: resultados del mismo orden).

---

## Limitaciones conocidas

1. **Fase A.** El descenso uniforme hasta `D` subdivide también los cubos INSIDE de
   nivel `< D`, que el monolítico deja gruesos salvo que el balance 2:1 lo obligue.
   El resultado sigue siendo un octree válido (balance verificado) pero más refinado
   en el interior. El programa lo **detecta** (contador en GPU) e imprime un aviso con
   el número de cubos INSIDE subdivididos; `--comparar` pasa entonces a comparar por
   **cobertura** (hojas BORDER idénticas, cada INSIDE contenida en una INSIDE del
   monolítico y mismo volumen INSIDE). En cortex no ocurre con `SPLIT` 1 y 2 (los 8
   cubos del nivel 1 son BORDER); una esfera de radio 0,95 con `D=4` subdivide 144 cubos
   INSIDE y deja 61.760 hojas INSIDE contra 61.704 del monolítico.
2. **Sin VTK con `SPLIT>0`.** Para exportar hay que usar `SPLIT=0`, que necesita el
   arreglo completo de hojas en VRAM.
3. **Tiempo ~2× en adaptativo** por las dos pasadas (la 1.ª descarta su DAG): nivel 11,
   14,65 s monolítico contra ~29 s por subárboles. Domina la clasificación.
4. **Solo espejos**, sin rotaciones ni permutaciones de ejes: 8 de las 48 simetrías del
   cubo. La esfera aprovecha 8× de un posible ~48×.
5. Los subárboles se procesan en secuencia en el host (solo índices); `SPLIT=D` fija
   hasta 8^D subárboles y el subárbol más grande debe caber en VRAM con su refinamiento
   (p. ej. uniforme nivel 11 necesita `SPLIT=3`).
6. Índice de 27 bits (~134 M nodos únicos por nivel) y enteros de 32 bits para el
   conteo de hojas por subárbol (el total se acumula en 64 bits por multiplicidad).
7. No se comparte entre niveles distintos ni se integra con la salida VTK en streaming
   de `experimental/`.

## Estructura

```text
src/dag.cuh, dag.cu, dag_global.inc   formato, dagLookup, dagBuild, DAG global incremental,
                                      conteos por multiplicidad, validaciones, cobertura
src/dag_subarboles.cuh/.cu            fases A/B/C, halo, cáscaras, pasadas, perfil
src/octree.cu, octree_halo.inc        refinamiento del base + halo, poda in-place, cáscaras
src/similitud.cuh/.cu                 medición de similitud aproximada (--similitud)
src/main.cu                           CLI, camino monolítico y por subárboles
tests/dag_test.cu                     pruebas (make test)
```
