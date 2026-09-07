# Comparación base vs experimental — datos medidos

Máquina: RTX 4050 Laptop (5,99 GiB VRAM), 16 GB RAM, WSL2, CUDA 12.0, `sm_89`.
Modelo: `cortex.mdl` (3152 vértices, 6300 triángulos).
Fecha de las mediciones: 2026-09-07.

- **base** = `../../octree` (proyecto de `../../src`)
- **exp** = `../octree` (este proyecto, `SPLIT=1`, 8 subárboles)

Los logs crudos que respaldan cada fila están en `raw/`.

---

## Corrección: ¿produce el mismo octree?

Comparación celda por celda del VTK: mismo conjunto de hexaedros, mismo `level`
y mismo `state`. El orden difiere (subárbol-mayor vs Morton-mayor), por eso la
comparación ordena antes de comparar.

| nivel | modo | hojas base | hojas exp | idénticos | pasadas |
|---|---|---|---|---|---|
| 3 | adaptativo | 252 | 252 | sí | 1 |
| 3 | uniforme | 259 | 259 | sí | 1 |
| 4 | adaptativo | 1.173 | 1.173 | sí | 1 |
| 4 | uniforme | 1.572 | 1.572 | sí | 1 |
| 5 | adaptativo | 5.501 | 5.501 | sí | 2 |
| 5 | uniforme | 10.646 | 10.646 | sí | 1 |
| 6 | adaptativo | 24.917 | 24.917 | sí | 2 |
| 6 | uniforme | 78.019 | 78.019 | sí | 1 |
| 7 | adaptativo | 105.634 | 105.634 | sí | 2 |
| 7 | uniforme | 596.068 | 596.068 | sí | 1 |
| 8 | adaptativo | 435.534 | 435.534 | sí | 2 |
| 8 | uniforme | 4.653.818 | 4.653.818 | sí | 1 |
| 9 | adaptativo | 1.774.809 | 1.774.809 | sí | 2 |
| 10 | adaptativo | 7.161.958 | 7.161.958 | sí | 2 |
| 11 | adaptativo | 28.786.873 | 28.786.873 | sí (tabla completa por nivel) | 2 |
| 12 | adaptativo | — (OOM) | 115.425.336 | el base no termina | 2 |

Fuente: `raw/02_validacion_niveles_3-8_DESPUES_del_fix.log`,
`raw/03_niveles_9-10_memoria_y_tiempo.log`, `raw/04_niveles_11-12.log`.

La verificación 2:1 sobre las fronteras cierra en **0 violaciones** en todas las
corridas adaptativas convergidas.

### Nivel 11 — tabla por nivel, ambos proyectos

Las tres columnas (hojas / dentro / borde) coinciden en los 11 niveles:

| nivel | hojas | dentro | borde |
|---|---|---|---|
| 1 | 8 | 0 | 8 |
| 2 | 55 | 1 | 54 |
| 3 | 252 | 50 | 202 |
| 4 | 1.173 | 325 | 848 |
| 5 | 5.501 | 2.029 | 3.472 |
| 6 | 24.917 | 10.924 | 13.993 |
| 7 | 105.634 | 49.252 | 56.382 |
| 8 | 435.534 | 209.824 | 225.710 |
| 9 | 1.774.809 | 871.307 | 903.502 |
| 10 | 7.161.958 | 3.546.955 | 3.615.003 |
| 11 | 28.786.873 | 14.324.530 | 14.462.343 |

---

## Memoria y tiempo

"Pico calculado" es el modelo del algoritmo (`Pico calculado del algoritmo`);
"pico medido" es lo que reportó el driver descontando el contexto CUDA.

| nivel | pico base | pico exp | factor | t base | t exp | factor |
|---|---|---|---|---|---|---|
| 9 | 204,24 MB | 33,43 MB | 6,1× | 0,588 s | 2,087 s | 3,5× |
| 10 | 821,96 MB | 121,03 MB | 6,8× | 2,037 s | 6,373 s | 3,1× |
| 11 | 3,22 GB | 464,72 MB | 7,1× | 7,696 s | 15,541 s | 2,0× |
| 12 | ~12,9 GB → **OOM** | **1,77 GB** | — | no termina | 65,885 s | — |

Picos medidos por el driver: nivel 9 → 212 MB base / 88 MB exp; nivel 10 →
828 MB base / 178 MB exp; nivel 12 → 1,79 GB exp (contra 1,77 GB calculados).

El factor de memoria supera el 8× ingenuo de dividir en 8 porque se suman dos
efectos: el working set baja a ~1/8 **y** la poda con reserva exacta elimina el
factor 2 del peor caso que tenía el base.

El factor de tiempo baja al crecer el problema (3,5× → 2,0×) porque los
lanzamientos por subárbol dejan de estar dominados por el overhead. Tiende al
2× estructural de las dos pasadas.

### Nivel 12 — el caso que el base no puede

Presupuesto del base en la etapa 12, que es donde falla con
`CUDA error en src/octree.cu:316 — out of memory`:

| buffer | tamaño |
|---|---|
| entrada (28.786.873 hojas × 48 B) | 1,29 GiB |
| scratch adaptativo (24 B/hoja) | 0,64 GiB |
| salida refinada (144.386.903 × 48 B) | 6,45 GiB |
| buffer de poda (peor caso) | 6,45 GiB |
| **pico** | **~12,9 GiB** contra 5,99 GiB de la tarjeta |

Resultado del experimental en el mismo nivel:

```
Subarboles procesados         : 8 (nivel 1)
Pasadas externas              : 2 de 3
Balance 2:1 entre subarboles  : verificado, 0 violaciones
Halo residente en VRAM        : 33.93 MB

    11 |   28786873 |   14324530 |   14462343 |   1.29 GB |  488.39 MB |   6.274
    12 |  115425336 |   57572261 |   57853075 |   5.16 GB |    1.77 GB |  24.914

  Pico calculado del algoritmo  : 1.77 GB
  Pico medido por el driver     : 1.79 GB
  Tiempo total de refinamiento  : 65.885 s
```

Medido con `--no-vtk`: el VTK de 115.425.336 cubos pesaría ~16,6 GB en disco.

---

## El fallo que se encontró durante la validación

`raw/01_validacion_niveles_3-8_ANTES_del_fix.log` es la corrida que destapó el
problema. La cáscara transportaba las celdas pero no la bandera `split`, y eso
cortaba la propagación en la frontera:

| nivel | base | exp (sin la bandera) | déficit |
|---|---|---|---|
| 7 adaptativo | 105.634 | 105.592 | −42 |
| 8 adaptativo | 435.534 | 435.275 | −259 |

Con `--passes 8` las violaciones se estancaban en 101 desde la segunda pasada:
convergía, pero a un punto fijo incorrecto. Ver la sección "Fase B" del
`../README.md`.

---

## Costo de saltarse el cierre de frontera (`--passes 1`)

Nivel 7, misma máquina:

| | `--passes 1` | `--passes 3` |
|---|---|---|
| hojas | 105.186 | 105.634 |
| dentro | 48.804 | 49.252 |
| borde | 56.382 | 56.382 |
| violaciones 2:1 | 2341 sin resolver | 0, verificado |
| pasadas | 1 | 2 |
| tiempo | 0,252 s | 0,438 s |

El conteo `Borde` es idéntico porque el criterio geométrico es local. Las 448
hojas que faltan son todas DENTRO: celdas interiores que solo se dividirían por
obligación del balance 2:1 a través de los planos que separan los octantes. La
divergencia arranca en el nivel 5 (5.466 contra 5.501) y se arrastra hacia
arriba.

Violaciones que deja la primera pasada por sí sola: 192 (nivel 5), 508 (6),
2341 (7), 6318 (8).

---

## Nivel 13 — SPLIT=1, corrida completa (usuario, GPU real)

Ejecutada por el usuario con `make run LEVEL=13`. Log completo en
`raw/05_nivel_13_split1_usuario.log`.

```
Subarboles procesados         : 8 (nivel 1)
Pasadas externas              : 2 de 3
Balance 2:1 entre subarboles  : verificado, 0 violaciones
Halo residente en VRAM        : 69.04 MB

    13 |  462261737 |  230840319 |  231421418 |  20.66 GB |    7.01 GB |   106.855

Pico calculado del algoritmo  : 7.01 GB
Pico medido por el driver     : 4.96 GB
VRAM total del dispositivo    : 6.00 GB
Tiempo total de refinamiento  : 724.136 s
```

**462.261.737 hojas, converge en 2 pasadas, 0 violaciones** — igual que en
todos los niveles anteriores con SPLIT=1. El VTK no se escribió: 462M cubos
superan el límite del formato legacy (268.435.455, por los índices de punto en
`int32`); el programa lo detectó y avisó en vez de escribir un archivo
corrupto, que es el comportamiento diseñado (ver "Límites conocidos" en
`../README.md`).

Dos observaciones de esta corrida real que no se veían en las mediciones de
niveles más bajos:

- **El nivel 13 con SPLIT=1 sí desborda a RAM del sistema, confirmado
  directamente.** Se agregó `gpuRequireBudget()`: antes de cada reserva
  grande consulta `cudaMemGetInfo` en el instante real y compara contra el
  total físico (6,00 GB), terminando el programa con `exit(1)` si no alcanza,
  en vez de dejar que el driver de Windows (WDDM) desborde en silencio a RAM
  del sistema. Al aplicarlo, **el nivel 13 dispara el guardián en la poda del
  primer subárbol de la primera pasada**:

  ```
  Error: poda (salida) necesita 2.78 GB adicionales, pero solo hay 1.47 GB
  libres de 6.00 GB de VRAM fisica.
  ```

  Log completo en `raw/08_nivel_13_con_guardian.log`.

  Esto **corrige dos afirmaciones anteriores de este mismo documento**, en
  direcciones opuestas — vale la pena dejar registrado el error de
  razonamiento, no solo el resultado final:
  1. Primero se especuló (sin verificar) que hubo desborde a RAM bajo WDDM.
  2. Después se corrigió eso citando que "Pico medido por el driver" (4,96
     GB) quedó bajo los 6,00 GB, concluyendo que probablemente NO hubo
     desborde — esa conclusión también era incorrecta.

  La razón de la contradicción: `gpuSampleVram()` solo muestrea en puntos
  fijos del código (después de generar la salida del refinamiento, después
  de reservar el buffer de poda), no en el instante exacto de mayor uso. El
  pico real ocurre un momento antes, cuando coexisten el arreglo recién
  refinado y el intento de reservar el buffer de poda — instante que
  `gpuRequireBudget()` sí intercepta porque corre justo ahí, no en un punto
  de muestreo aparte. **Conclusión que sí queda firme, verificada
  directamente**: el nivel 13 con SPLIT=1 no vive completo en VRAM física;
  las corridas "exitosas" anteriores (usuario y reproducción) dependían de
  desborde a RAM del sistema sin que el programa lo reportara. Bajo la
  política de que esto debe tratarse como error, **el nivel 13 con SPLIT=1 no
  se puede dar por logrado tal cual está** — hace falta `SPLIT=2` o más para
  que quepa en VRAM real, y `SPLIT=2` todavía no converge correctamente (ver
  más abajo).
- **El tiempo por subárbol varía fuerte** (17,3 s a 110,4 s en la pasada 1):
  el cortex no está centrado en el cubo raíz, así que unos octantes cargan
  mucha más superficie de la corteza que otros. Es el desbalance de 1,5–2×
  entre octantes anticipado en el diseño original.

---

## Convergencia del balance 2:1 según el nivel de descomposición

Medido en el nivel 11, con `--passes 3` (el default) en los tres casos.

| SPLIT | subárboles | hojas | ¿converge? | violaciones sin resolver | pico calc. |
|---|---|---|---|---|---|
| 1 | 8 | 28.786.873 (correcto) | sí, en 2 pasadas | 0 | 464,7 MB |
| 2 | 55 | 28.786.537 | **no** | 4163 | 204,5 MB |
| 3 | 259 | 28.779.845 | **no** | 83.184 | 199,5 MB |

**Con más granularidad, `--passes 3` no basta.** El halo de una pasada viene
de la pasada anterior, así que una marca se propaga un subárbol por pasada: el
número de pasadas necesario escala con el diámetro del grafo de adyacencia
entre subárboles, no es una constante. Con 8 octantes (SPLIT=1) todos se tocan
entre sí y un salto alcanza. Con 55 o 259 subárboles en rejilla, una cadena de
propagación puede necesitar varios saltos más, y subir el split sin subir
`--passes` da un resultado **incorrecto sin avisar tanto como debería** (el
programa sí imprime "violaciones SIN resolver", pero es fácil no mirarlo).

Pendiente: medir si `--passes 8` o más cierra en SPLIT=2/3, y si el costo
extra de pasadas sigue compensando la memoria ahorrada frente a subir
`--chunk`/aceptar el pico de SPLIT=1. Hasta entonces, **SPLIT=2/3 no está
validado como correcto** — solo SPLIT=1 lo está, hasta el nivel 13.

---

## Cómo reproducir

```bash
# corrección: comparar celda por celda contra el base
cd ..
make -C .. && make
../octree 8 cortex.mdl --adaptive --quiet
./octree   8 cortex.mdl --adaptive --quiet
# comparar output/octree_pruned_level_8.vtk de ambos

# memoria y tiempo
../octree 11 cortex.mdl --adaptive --quiet   # base
./octree   11 cortex.mdl --adaptive --quiet --no-vtk

# el nivel que el base no alcanza
./octree 12 cortex.mdl --adaptive --no-vtk
```
