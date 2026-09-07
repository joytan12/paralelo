# Octree CUDA — experimental: descomposición por subárboles

Variante del proyecto GPU de `../src` con **un solo cambio de arquitectura**:
en vez de mantener todo el octree en un único arreglo de VRAM, la raíz se
divide en subárboles y la GPU los procesa **uno a la vez**, evacuando cada uno
a disco apenas termina.

Todo el cómputo sigue viviendo en la GPU. La CPU no toca un solo cubo: solo
hace `fwrite` de bytes que la GPU ya generó y el DMA bajó.

El resultado es **el mismo octree** que produce el proyecto base — mismas
hojas, mismos niveles, mismos estados. Solo cambia el orden en que se escriben
al archivo.

---

## Por qué

El proyecto base revienta en el nivel 12 sobre una RTX 4050 Laptop (5,99 GiB):

| Buffer (etapa 12) | Tamaño |
|---|---|
| entrada (28.786.873 hojas × 48 B) | 1,29 GiB |
| scratch adaptativo (24 B/hoja) | 0,64 GiB |
| salida refinada (144.386.903 × 48 B) | 6,45 GiB |
| buffer de poda (peor caso) | 6,45 GiB |
| **pico** | **~12,9 GiB** |

Una sola copia del arreglo de nivel 12 ya no cabe en la tarjeta. En WSL2 el
driver pagina a RAM del sistema bajo WDDM, así que la etapa 11 ya estaba
haciendo thrashing sobre PCIe antes de que el `cudaMalloc` de la poda fallara.

Con la descomposición el working set baja a ~1/N del original y el resultado
nunca se acumula: sale a disco subárbol por subárbol.

---

## Cómo funciona

### Fase A — descenso uniforme hasta el nivel de descomposición

Se refina la raíz uniformemente `D` niveles (`--split-level D`, por defecto 1),
se clasifica y se poda. Quedan a lo más `8^D` hojas, cada una raíz de un
subárbol.

Esto no es cosmético: garantiza que **toda hoja posterior tenga un dueño
único** (su ancestro de nivel `D`). Sin ello, una hoja INSIDE que sobrevivió en
el nivel 1 sería ancestro de varios subárboles y no pertenecería a ninguno. Con
`D=1` el resultado es idéntico al del base, porque el base también divide la
raíz completa en el nivel 1.

### Fase B — cada subárbol, solo, con halo

Cada subárbol se refina del nivel `D` al máximo sin salir de la GPU. El
problema es que el balance 2:1 **es una restricción global**: el kernel
`markCoarserNeighborsKernel` busca vecinos más gruesos por búsqueda binaria
sobre todas las hojas. Con solo el subárbol en VRAM, los vecinos que viven en
un hermano son invisibles y el balance queda roto en las caras internas.

La solución es el **halo (ghosts)**. La lista de trabajo del refinamiento pasa
a ser la concatenación lógica

```
[ hojas propias | cáscaras de los demás subárboles ]
```

resuelta con un accesor (`fetchCube`), sin copiar las propias. Los ghosts:

- participan en el marcado y en la búsqueda de vecinos, así que un vecino de
  otro subárbol se encuentra igual que uno propio;
- **nunca se materializan**: `generateAdaptiveOutputKernel` solo recorre la
  porción propia.

La **cáscara** de un subárbol son sus hojas que tocan su frontera exterior: una
hoja de nivel `L` con coordenadas locales en un extremo de `[0, 2^(L-D)-1]`.
Son las únicas que un vecino puede llegar a ver, y son O(superficie), no
O(volumen) — decenas de KB frente a los GB del subárbol.

Dos detalles hacen que esto sea correcto y no una aproximación:

**1. La cáscara lleva la bandera `split`, no solo la celda.** Una hoja de
frontera de A puede estar marcada para dividirse por el balance *interno* de A,
disparada por una celda fina del interior de A que no vive en la cáscara. Si B
solo recibe la celda y recalcula la marca desde el predicado geométrico
(`state == BORDER && level < max_level`), obtiene 0 y nunca se entera. La
cadena `celda fina interior de A -> hoja de frontera de A -> hoja gruesa de B`
se corta justo en la frontera.

Se detectó exactamente así: sin la bandera, la verificación 2:1 se estancaba en
101 violaciones a partir de la segunda pasada — un punto fijo, pero el
equivocado, y el nivel 7 daba 105.592 hojas contra las 105.634 del base. Con la
bandera converge a 0 y el resultado vuelve a coincidir hoja por hoja. Por eso
`launchAdaptiveRefinement` devuelve `d_split_own` y `launchExtractShell` filtra
celdas y banderas con el mismo stencil.

El halo se arma solo con los subárboles **adyacentes** (cajas de nivel `D` cuyos
índices de rejilla difieren a lo más en 1 por eje). Con `D=1` los 8 octantes se
tocan todos y no cambia nada; con `D=2` cada subárbol pasa de 63 vecinos
potenciales a 26 como máximo, que es lo que hace practicable bajar la
granularidad para llegar más profundo.

**2. Las cáscaras se guardan por paso de refinamiento, no solo la final.** El
halo que el subárbol `s` usa en el paso `L` es la cáscara de sus vecinos *en
ese mismo paso `L`*. Ese lockstep es esencial: el algoritmo original avanza
todos los niveles a la vez, y la regla "si yo me divido, mi vecino más grueso
también" solo tiene sentido si ambos lados están en el mismo nivel.

### Fase C — cierre y verificación

En la primera pasada el halo está vacío (nadie ha sido procesado todavía), así
que puede quedar algún par en violación en las fronteras. Al terminar la
pasada se ejecuta `launchVerifyBalance` sobre la **unión de las cáscaras
finales**: para cada hoja se recorren sus 26 vecinos de su mismo nivel y se
busca, subiendo por los ancestros, si el que los cubre es una hoja dos o más
niveles más gruesa.

- Si hay **0 violaciones**, el resultado es un punto fijo del mismo sistema de
  marcas que resuelve el algoritmo global: es correcto y ya está escrito.
- Si hay violaciones, se repite la fase B usando las cáscaras recién medidas
  como halo. El proceso es monótono (las marcas solo crecen) y converge.

Dentro de un subárbol el balance lo garantiza el algoritmo original sin
cambios, así que la verificación solo necesita mirar las cáscaras: es
exactamente lo que la descomposición podría haber roto.

En la práctica converge en **2 pasadas** (medido en los niveles 5 a 8: la
primera deja entre 192 y 6318 violaciones, la segunda cierra en 0). En modo
`--uniform` no hay búsqueda de vecinos, los subárboles son independientes por
construcción y basta 1.

### Evacuación en streaming

`vtk_stream.cu` reemplaza al `vtk_writer.cpp` del base, que juntaba todos los
cubos en un `malloc` de host (5,2 GiB en el nivel 12) antes de escribir.

Aquí:

- kernels por sección (`POINTS`, `CELLS`, `CELL_TYPES`, `level`, `state`)
  generan los bytes del archivo **ya en big-endian, en la GPU**;
- se bajan por trozos con `cudaMemcpyAsync` a un anillo de dos buffers pinned;
- mientras la CPU hace `fwrite` del trozo `i-2`, el DMA baja el `i-1` y la GPU
  genera el `i`.

Pico de RAM de host: ~50 MB (dos buffers), no gigabytes.

Como el formato VTK Legacy es seccionado y los contadores van en la cabecera
—que no se conoce hasta el final— cada sección se acumula en un temporal y al
cerrar se escribe la cabecera y se concatenan. El archivo resultante tiene
exactamente el mismo formato que el del proyecto base.

---

## Uso

Misma interfaz que el base, más cuatro opciones:

```
./octree [niveles] [modelo.mdl] [opciones]

  -q, --quiet             Sin prints intermedios
  -u, --uniform           Refinamiento uniforme
  -a, --adaptive          Adaptativo con balance 2:1 (por defecto)
  -s, --split-level N     8^N subárboles secuenciales. Default: 1
  -p, --passes N          Máximo de pasadas de cierre del balance. Default: 3
      --chunk N           Cubos por trozo de evacuación. Default: 262144
      --no-vtk            No escribe el archivo VTK
```

Por Makefile:

```bash
make run LEVEL=8
make run LEVEL=12 SPLIT=1 QUIET=1
make run LEVEL=13 SPLIT=1 VTK=0
```

`SPLIT` es el dial de memoria: subirlo divide el working set por 8^D. **Con
SPLIT=1 el resultado es correcto y está validado hasta el nivel 13** (462M
hojas, converge en 2 pasadas, 0 violaciones — ver `bench/BENCHMARK.md`).

**`SPLIT=2` y `SPLIT=3` no están validados como correctos con `PASSES` por
defecto.** El halo de una pasada viene de la pasada anterior, así que una
marca se propaga un subárbol por pasada, y el número de pasadas necesario
escala con el diámetro del grafo de adyacencia entre subárboles — no es una
constante. Con 8 octantes (SPLIT=1) todos se tocan entre sí y bastan 2. Medido
en el nivel 11 con `PASSES=3`: SPLIT=2 (55 subárboles) deja 4163 violaciones
sin resolver, SPLIT=3 (259 subárboles) deja 83.184. Sube `PASSES` si usas
`SPLIT>1` y revisa siempre la línea "Balance 2:1 entre subárboles" en la
salida antes de confiar en el resultado.

`PASSES=1` acepta la primera pasada aunque no converja — es más rápido, pero
el programa avisa cuántas violaciones quedaron.

---

## Diferencias con el proyecto base

| | base | experimental |
|---|---|---|
| Estructura del bucle | nivel-externo, un arreglo global | subárbol-externo, uno a la vez |
| Balance 2:1 en fronteras | trivial (todo está en VRAM) | halo de cáscaras + pasadas de cierre |
| Poda | reserva el peor caso (`2 × n`) | cuenta primero, reserva el tamaño exacto |
| Cierre del balance | siempre `max_level` rondas | corta al alcanzar el punto fijo |
| Salida | `malloc` de host con todo el octree | streaming VRAM → pinned → disco |
| RAM de host en el pico | `n_hojas × 48 B` | ~50 MB constantes |
| Verificación 2:1 | no hay | kernel sobre las cáscaras, reportado al final |

Las dos últimas filas de la mitad inferior (poda exacta y corte del cierre) no
son parte de la descomposición: son mejoras locales que se aprovecharon porque
el factor 2 de la poda decide si `SPLIT=1` alcanza o hace falta `SPLIT=2`.
Ninguna cambia el resultado.

**El orden de las celdas en el VTK es distinto** (subárbol-mayor en vez de
Morton-mayor). El conjunto de celdas, sus niveles y sus estados son idénticos;
verificado hoja por hoja contra el base.

---

## Resultados medidos

RTX 4050 Laptop (5,99 GiB), WSL2, `cortex.mdl` (3152 vértices, 6300 triángulos),
modo adaptativo, `SPLIT=1`.

### Corrección

Comparado hoja por hoja contra `../octree`: mismo conjunto de hexaedros, mismo
`level` y mismo `state` en el VTK.

| nivel | modo | hojas | ¿idéntico al base? | pasadas |
|---|---|---|---|---|
| 3–8 | adaptativo | 252 … 435.534 | sí (6/6) | 1–2 |
| 3–8 | uniforme | 259 … 4.653.818 | sí (6/6) | 1 |
| 9 | adaptativo | 1.774.809 | sí | 2 |
| 10 | adaptativo | 7.161.958 | sí | 2 |
| 11 | adaptativo | 28.786.873 | sí (tabla completa por nivel) | 2 |
| 12 | adaptativo | 115.425.336 | el base no termina | 2 |

La verificación 2:1 sobre las fronteras cierra en **0 violaciones** en todas las
corridas adaptativas.

### Memoria y tiempo

| nivel | pico base | pico experimental | factor | t base | t exp |
|---|---|---|---|---|---|
| 9 | 204,2 MB | 33,4 MB | 6,1× | 0,59 s | 2,09 s |
| 10 | 822,0 MB | 121,0 MB | 6,8× | 2,04 s | 6,37 s |
| 11 | 3,22 GB | 464,7 MB | 7,1× | 7,70 s | 15,54 s |
| 12 | ~12,9 GB → OOM | **1,77 GB** | — | no termina | 65,9 s |

En el nivel 12 el pico medido por el driver fue 1,79 GB contra 1,77 GB
calculados, y el halo residente ocupó 33,9 MB. Queda margen de sobra en la
tarjeta; para el nivel 13 el dial es `SPLIT=2`.

El factor de reducción es mayor que el 8× ingenuo de dividir en 8 porque se
suman dos efectos: el working set baja a ~1/8, y la poda con reserva exacta
elimina el factor 2 del peor caso que tenía el base.

### El costo

**Entre 2× y 3× más lento.** Dos causas: la fase B se ejecuta dos veces (la
primera pasada construye las cáscaras, la segunda las usa y confirma), y los
lanzamientos por subárbol son 8 veces más pequeños. La segunda causa se
amortiza al crecer el problema — de 3,5× en el nivel 9 a 2,0× en el nivel 11 —
así que el costo tiende al 2× estructural de las dos pasadas.

`--passes 1` salta la segunda pasada y recupera ese 2×, pero el programa
reporta cuántas violaciones quedaron sin cerrar en vez de esconderlas. Para
referencia, la primera pasada por sí sola deja entre 192 (nivel 5) y 6318
(nivel 8) violaciones de frontera.

---

## Límites conocidos

- El VTK legacy escribe índices de punto como `int32` y hay 8 puntos por cubo,
  así que el formato tope en 268.435.455 cubos. El programa lo detecta, avisa y
  no escribe un archivo corrupto.
- El VTK con 8 puntos propios por cubo pesa ~144 B/cubo: el nivel 12 son
  115.425.336 cubos ≈ **16,6 GB** en disco. El formato lo aguanta (el tope son
  268.435.455 cubos) pero el disco quizá no: usa `--no-vtk` si solo te interesan
  las métricas. Las mediciones del nivel 12 de arriba se tomaron así.
- Los contadores de hojas siguen siendo `int`. Por encima de ~240M hojas hay
  que pasarlos a `long long` (`n_input * 8` en el refinamiento uniforme es el
  primero en desbordar).
- `Cube` sigue ocupando 48 B. `center`, `half_size`, `level` y `grid_index` son
  todos derivables de `path_code`: una representación compacta de 9 B daría
  otro 5,3× de margen, ortogonal a la descomposición.
