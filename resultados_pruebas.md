# Informe comparativo de pruebas CPU, GPU y external/MixedOcTree

**Fecha:** 7 de octubre de 2026  
**Modelo:** `cortex.mdl` — 3.152 vértices y 6.300 triángulos  
**Modos ejercitados:** adaptativo 2:1, uniforme y external/MixedOcTree (superficie / dominio completo)  
**Objetivo:** avanzar niveles sucesivos hasta el límite de memoria del refinamiento.

## Resumen

- Las versiones **CPU single** y **CPU multi** ejecutaron ambos modos. Los procesos recorren los niveles de forma acumulativa; los límites de tiempo impidieron que CPU single alcanzara OOM en cualquiera de sus modos. CPU multi adaptativo también quedó limitado por tiempo. CPU multi uniforme sí alcanzó el límite de asignación en nivel 10 con un límite de espacio virtual de 5 GiB.
- **external/MixedOcTree** se compiló y ejecutó en sus modos de refinamiento de superficie (`-s`) y de dominio completo (`-a`). Con un límite de 5 GiB, superficie llegó al nivel 8 y dominio completo al nivel 6 antes de `std::bad_alloc`.
- Las tres variantes CUDA compilaron para `sm_89`, pero ninguna pudo inicializar CUDA en este entorno. El runtime informó `CUDA driver version is insufficient for CUDA runtime version`; `nvidia-smi` también indicó que el acceso a la GPU está bloqueado por el sistema operativo. No se pudieron ejecutar niveles GPU.
- La comprobación de VTK entre CPU single y CPU multi coincidió byte por byte en nivel 5, en ambos modos.
- **Hallazgo de corrección:** los dos ejecutables CPU coinciden entre sí, pero en adaptativo se apartan de los conteos de nivel 7 y 8 publicados en los datos de referencia del repositorio. La diferencia aparece solo en hojas `INSIDE`; conviene revisarla antes de usar estos resultados como validación cruzada con CUDA.

## Entorno y controles

| Elemento | Valor observado |
|---|---|
| Sistema | WSL2, kernel Linux `6.6.87.2-microsoft-standard-WSL2` |
| CPU visible | 20 procesadores lógicos; OpenMP configurado con 20 hilos |
| RAM / swap visibles | 7,5 GiB / 2 GiB |
| CUDA Toolkit | 12.0.140 (`nvcc`) |
| Arquitectura de compilación CUDA | `sm_89` |
| Acceso a GPU | Bloqueado: `nvidia-smi` no inicializa NVML; el runtime CUDA informa driver insuficiente |
| Límite de memoria en corridas CPU largas | `ulimit -v 5242880` KiB (5 GiB de espacio virtual por proceso) |
| Límite inicial por corrida CPU | 120 s; CPU multi uniforme tuvo además una corrida extendida de 600 s |
| external/MixedOcTree | Compilación C++11 `-O2` con las 47 fuentes de `external/MixedOcTree/src/CMakeLists.txt`; límite por proceso de 5 GiB y timeout de 600 s |

Las ejecuciones se hicieron desde directorios temporales bajo `/tmp/paralelo_bench_20261007`, para no sobrescribir los VTK existentes del repositorio. Los logs de salida quedaron en [`benchmark_logs/`](benchmark_logs/).

## Vista comparativa común en nivel solicitado 5

| Implementación / modo | Resultado | Tiempo reportado | Memoria reportada |
|---|---:|---:|---:|
| CPU single / adaptativo | 5.501 hojas | 0,465 s de refinamiento | pico estimado 0,45 MB |
| CPU multi / adaptativo, 20 hilos | 5.501 hojas | 0,095 s de refinamiento | pico estimado 0,46 MB |
| CPU single / uniforme | 10.646 hojas | 0,879 s de refinamiento | pico estimado 0,68 MB |
| CPU multi / uniforme, 20 hilos | 10.646 hojas | 0,169 s de refinamiento | pico estimado 0,74 MB |
| external/MixedOcTree / superficie (`-s 5`) | 26.209 elementos mixtos | 0,364 s generación; 0,382 s total | RSS máximo 34 MB |
| external/MixedOcTree / dominio (`-a 5`) | 29.329 elementos mixtos | 0,430 s generación; 0,463 s total | RSS máximo 52 MB |
| CUDA base / experimental / experimental2 | Sin resultado de nivel | No ejecutable: driver CUDA inaccesible | Sin medición actual de VRAM |

Esta tabla reúne métricas útiles para orientar la comparación, pero los conteos de hojas CPU y elementos mixtos externos no representan la misma salida. Los tiempos CPU miden refinamiento; external informa generación/refinamiento y escritura por separado. La memoria CPU es la estimación interna del programa y la memoria external es RSS del proceso, por lo que tampoco es una medición homogénea.

## Resultados CPU hasta tiempo o memoria límite

Cada corrida larga solicitó `LEVEL=20`, por lo que cada log muestra las etapas completadas antes de detenerse. “Pico etapa” es la estimación del programa; `Maximum resident set size` es el máximo residente medido por `/usr/bin/time`.

| Versión / modo | Resultado alcanzado | Tiempo y memoria relevantes | Fin de corrida |
|---|---|---|---|
| CPU single / adaptativo | Nivel 8 completado: **435.289 hojas**. Nivel 9 no terminó. | Nivel 8: 108,643 s; pico estimado 25,29 MB; RSS máx. 26.768 KiB. | Límite de 120 s (`timeout`, código 124), durante nivel 9. |
| CPU multi / adaptativo, 20 hilos | Nivel 8 completado: **435.289 hojas**. Nivel 9 no terminó. | Nivel 8: 90,518 s; pico estimado 26,86 MB; RSS máx. 30.276 KiB. | Límite de 120 s (`timeout`, código 124), durante nivel 9. |
| CPU single / uniforme | Nivel 7 completado: **596.068 hojas**. En nivel 8 generó 4.768.544 cubos para clasificar, pero no terminó esa clasificación. | Nivel 7: 37,712 s; pico estimado 28,68 MB; RSS máx. 129.392 KiB. | Límite de 120 s (`timeout`, código 124), durante nivel 8. |
| CPU multi / uniforme, primera corrida | Nivel 8 completado: **4.653.818 hojas**. En nivel 9 generó 37.230.544 cubos, sin completar su clasificación dentro del límite inicial. | Nivel 8: 38,835 s; pico estimado 252,15 MB; RSS máx. 984.412 KiB. | Límite inicial de 120 s (`timeout`, código 124), durante nivel 9. |
| CPU multi / uniforme, corrida extendida, 20 hilos | Nivel 9 completado: **36.775.618 hojas**. El refinamiento de nivel 10 falló al asignar memoria. | Nivel 9: 330,775 s; pico estimado 1.978,02 MB; RSS máx. 2.028.396 KiB. Tiempo total: 6:32,47. | SIGSEGV / código 139 en nivel 10, con el límite de 5 GiB de espacio virtual. |

En la última corrida, nivel 10 necesitaba crear 8 hijos para cada una de 36.775.618 hojas. Con `sizeof(Cube)=24 B` en CPU, el arreglo de salida por sí solo requería aproximadamente **6,58 GiB**, por encima del límite de proceso. El código no comprueba el resultado de `malloc` antes de usar el puntero; por eso el fallo aparece como señal 11 y no como un error de memoria controlado. Esta es una prueba bajo un límite de espacio virtual explícito, no un OOM de toda la máquina.

Las corridas adaptativas quedaron limitadas por tiempo antes de alcanzar memoria alta. El código CPU busca vecinos con bucles anidados sobre las hojas durante el cierre del balance [CPU single](cpu_single/src/octree.cpp#L49) y [CPU multi](cpu_multi/src/octree.cpp#L40); en multi ese tramo también es secuencial. Esto explica por qué el tiempo puede impedir llegar al umbral de memoria.

### Etapas completas por nivel

Los tiempos son los de cada etapa reportados por el programa. El pico calculado es `single / multi` en MB. En modo adaptativo, ambos ejecutables completaron hasta nivel 8; en uniforme, CPU single completó hasta nivel 7, mientras CPU multi alcanzó nivel 9 en la corrida extendida.

**Adaptativo**

| Nivel | Hojas | Dentro | Borde | Tiempo single / multi (s) | Pico single / multi (MB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0 | 8 | 0,000 / 0,001 | 0,11 / 0,11 |
| 2 | 55 | 1 | 54 | 0,002 / 0,002 | 0,11 / 0,11 |
| 3 | 252 | 50 | 202 | 0,031 / 0,013 | 0,13 / 0,13 |
| 4 | 1.173 | 325 | 848 | 0,084 / 0,015 | 0,18 / 0,19 |
| 5 | 5.501 | 2.029 | 3.472 | 0,365 / 0,053 | 0,45 / 0,46 |
| 6 | 24.917 | 10.924 | 13.993 | 1,644 / 0,339 | 1,58 / 1,66 |
| 7 | 105.620 | 49.238 | 56.382 | 9,023 / 4,524 | 6,23 / 6,61 |
| 8 | 435.289 | 209.579 | 225.710 | 108,643 / 90,518 | 25,29 / 26,86 |

**Uniforme**

| Nivel | Hojas | Dentro | Borde | Tiempo single / multi (s) | Pico single / multi (MB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0 | 8 | 0,000 / 0,001 | 0,11 / 0,11 |
| 2 | 55 | 1 | 54 | 0,002 / 0,003 | 0,11 / 0,11 |
| 3 | 259 | 57 | 202 | 0,027 / 0,008 | 0,13 / 0,13 |
| 4 | 1.572 | 724 | 848 | 0,123 / 0,014 | 0,20 / 0,21 |
| 5 | 10.646 | 7.174 | 3.472 | 0,750 / 0,078 | 0,68 / 0,74 |
| 6 | 78.019 | 64.026 | 13.993 | 5,157 / 0,616 | 4,01 / 4,49 |
| 7 | 596.068 | 539.686 | 56.382 | 37,712 / 4,724 | 28,68 / 32,80 |
| 8 | 4.653.818 | 4.428.108 | 225.710 | no completado / 38,835 | no completado / 252,15 |
| 9 | 36.775.618 | 35.872.116 | 903.502 | — / 330,775 | — / 1.978,02 |

## Comparación CPU single / CPU multi en nivel 5

Se ejecutaron los dos binarios a nivel 5 y se compararon los VTK resultantes con SHA-256. Los archivos coinciden exactamente entre CPU single y multi para cada modo.

| Modo | Hojas | Tiempo de refinamiento single | Tiempo de refinamiento multi (20 hilos) | SHA-256 común |
|---|---:|---:|---:|---|
| Adaptativo | 5.501 | 0,465 s | 0,095 s | `6af34dd2ee25be0fad54a2cf5e009d09effc03ab13b3109b307c5f07cbaf3307` |
| Uniforme | 10.646 | 0,879 s | 0,169 s | `bedc48649ec7ca6acae0a8fb899f8c09dff819221f4317aa627549cf9c90f2fb` |

Los tiempos corresponden a refinamiento y no incluyen la escritura final del VTK.

## Diferencia en los conteos adaptativos de nivel 7 y 8

Las corridas CPU single y multi produjeron los mismos conteos entre sí, pero no coinciden con la referencia CUDA/experimental registrada en el repositorio:

| Nivel | CPU single y multi, corridas actuales | Referencia del repositorio | Diferencia |
|---|---:|---:|---:|
| 7 adaptativo | 105.620 hojas (49.238 dentro, 56.382 borde) | 105.634 (49.252 dentro, 56.382 borde) | −14 hojas `INSIDE` |
| 8 adaptativo | 435.289 hojas (209.579 dentro, 225.710 borde) | 435.534 (209.824 dentro, 225.710 borde) | −245 hojas `INSIDE` |

La referencia está en [`experimental/bench/BENCHMARK.md`](experimental/bench/BENCHMARK.md#L45). Hasta nivel 6, los conteos CPU sí coinciden con esa tabla. Los conteos de borde siguen coincidiendo en nivel 7 y 8; la diferencia es de hojas interiores. No se determinó la causa en esta tanda. Requiere comparar la regla CPU de vecindad/balance con CUDA cuando haya GPU disponible.

Los logs completos de estas corridas son [`cpu_single_adaptive.log`](benchmark_logs/cpu_single_adaptive.log), [`cpu_multi_adaptive.log`](benchmark_logs/cpu_multi_adaptive.log), [`cpu_single_uniform.log`](benchmark_logs/cpu_single_uniform.log), [`cpu_multi_uniform_timeout120.log`](benchmark_logs/cpu_multi_uniform_timeout120.log) y [`cpu_multi_uniform_limit5g.log`](benchmark_logs/cpu_multi_uniform_limit5g.log). [`cpu_multi_uniform_memory.log`](benchmark_logs/cpu_multi_uniform_memory.log) es un ensayo preliminar interrumpido durante nivel 9; no se usa como resultado final.

## external/MixedOcTree

`external/MixedOcTree` genera una malla de elementos mixtos (no una lista equivalente de hojas cúbicas del octree interno). Se probaron `-s N` —refina octantes que intersectan la superficie— y `-a N` —refina todos los elementos del dominio— usando el mismo `cortex.mdl`. Por ello, los conteos de elementos y el nivel solicitado sirven para comparar coste y escalamiento, pero **no son equivalentes uno a uno** con hojas CPU internas.

El ejecutable se compiló en `/tmp` a partir de las fuentes declaradas en `external/MixedOcTree/src/CMakeLists.txt`, sin editar el subrepositorio. Las corridas largas pidieron nivel máximo 20, con timeout de 600 s y límite virtual de 5 GiB. En las corridas exitosas, “generación” es el tiempo reportado por el programa, que incluye generación y refinamiento de la malla; “total” suma además la escritura VTK. La memoria es el máximo RSS del proceso.

### Corridas de comparación en nivel solicitado 5

| Modo | Elementos de salida | Generación | Escritura | Total | RSS máximo |
|---|---:|---:|---:|---:|---:|
| Superficie (`-s 5`) | 26.209 | 364 ms | 17 ms | 382 ms | 34.808 KiB (34 MB) |
| Dominio completo (`-a 5`) | 29.329 | 430 ms | 32 ms | 463 ms | 53.644 KiB (52 MB) |

Para orientar una comparación de escala en el nivel solicitado 5, el octree interno produjo 5.501 hojas adaptativas y 10.646 uniformes; external produjo 26.209 elementos en modo superficie y 29.329 en modo dominio. Sus datos y tipos de elemento difieren, así que no se debe interpretar la razón entre conteos como una comparación de precisión o de trabajo idéntico. En tiempo total external registró 0,382 s (`-s`) y 0,463 s (`-a`); CPU single registró 0,465 s adaptativo y 0,879 s uniforme, mientras CPU multi registró 0,095 s y 0,169 s, respectivamente. Las definiciones de etapas también difieren: external incluye la generación y posprocesamiento en su tiempo de generación y separa la escritura.

### Escalamiento hasta el fallo de memoria

Los niveles siguientes son las etiquetas que imprime MixedOcTree. La tabla conserva el conteo de elementos y el tiempo de cada etapa completada.

**Refinamiento de superficie (`-s 20`)**

| Nivel impreso | Elementos | Tiempo de etapa (ms) | Balance (ms) |
|---:|---:|---:|---:|
| 0 | 24 | 5 | 0 |
| 1 | 104 | 7 | 0 |
| 2 | 478 | 13 | 0 |
| 3 | 2.083 | 32 | 0 |
| 4 | 8.741 | 103 | 4 |
| 5 | 36.063 | 389 | 34 |
| 6 | 146.179 | 1.565 | 189 |
| 7 | 588.308 | 6.224 | 966 |
| 8 | 2.361.096 | 28.159 | 10.371 |

Después de completar la etapa etiquetada 8, la asignación del refinamiento siguiente lanzó `std::bad_alloc`. RSS máximo: 5.138.208 KiB (aprox. 4,90 GiB); tiempo total hasta el fallo: 2:14,14; salida 134 (SIGABRT).

**Refinamiento de dominio completo (`-a 20`)**

| Nivel impreso | Elementos | Tiempo de etapa (ms) | Balance (ms) |
|---:|---:|---:|---:|
| 0 | 24 | 5 | 0 |
| 1 | 104 | 7 | 0 |
| 2 | 526 | 16 | 0 |
| 3 | 3.347 | 41 | 1 |
| 4 | 23.605 | 159 | 7 |
| 5 | 176.071 | 945 | 50 |
| 6 | 1.357.611 | 5.236 | 486 |

El refinamiento siguiente lanzó `std::bad_alloc`. RSS máximo: 5.240.060 KiB (aprox. 5,00 GiB); tiempo total hasta el fallo: 1:37,55; salida 134 (SIGABRT). En ambos casos se solicitó hasta nivel 20, así que el fallo de asignación —no el nivel objetivo ni el timeout de 600 s— detuvo la corrida.

Logs: [compilación](benchmark_logs/build_external.log), [superficie nivel 5](benchmark_logs/external_surface_l5.log), [dominio completo nivel 5](benchmark_logs/external_all_l5.log), [superficie hasta memoria](benchmark_logs/external_surface_limit5g.log) y [dominio completo hasta memoria](benchmark_logs/external_all_limit5g.log). El intento inicial de compilar indiscriminadamente todos los `.cpp` del directorio se conserva como antecedente en [build_external_all_cpp_attempt.log](benchmark_logs/build_external_all_cpp_attempt.log); el binario usado se construyó con el conjunto de fuentes de CMake.

## Compilación y ejecución CUDA

Las tres compilaciones terminaron correctamente con `nvcc 12.0` y `sm_89`:

- CUDA base: [`build_gpu_base.log`](benchmark_logs/build_gpu_base.log)
- CUDA experimental por subárboles: [`build_gpu_experimental.log`](benchmark_logs/build_gpu_experimental.log)
- CUDA experimental2, unión de superficies: [`build_gpu_experimental2.log`](benchmark_logs/build_gpu_experimental2.log)

Se intentó arrancar nivel 1 en cada variante. Las tres abortaron antes de crear el contexto o ejecutar kernels, con el mismo error de driver. El test de unión de `experimental2` también abortó en `cudaFree(0)` por el mismo motivo.

- [CUDA base, intento de ejecución](benchmark_logs/gpu_base_runtime.log)
- [CUDA experimental, intento de ejecución](benchmark_logs/gpu_experimental_runtime.log)
- [CUDA experimental2, intento de ejecución](benchmark_logs/gpu_experimental2_runtime.log)
- [Test de unión GPU de experimental2](benchmark_logs/gpu_experimental2_merge_test.log)

Por la falta de acceso al dispositivo no hay mediciones actuales de VRAM, tiempo ni nivel OOM para CUDA. La compilación correcta no demuestra que los kernels hayan corrido.

## Datos CUDA previos del repositorio (no medidos en esta sesión)

Como referencia separada, [`experimental/bench/BENCHMARK.md`](experimental/bench/BENCHMARK.md#L65) registra mediciones del 7 de septiembre de 2026 en una RTX 4050 Laptop con 5,99 GiB de VRAM y WSL2. En modo adaptativo y `SPLIT=1`, reporta:

| Nivel | Base: pico / tiempo | Experimental: pico / tiempo | Resultado |
|---|---:|---:|---|
| 9 | 204,24 MB / 0,588 s | 33,43 MB / 2,087 s | ambos completan |
| 10 | 821,96 MB / 2,037 s | 121,03 MB / 6,373 s | ambos completan |
| 11 | 3,22 GB / 7,696 s | 464,72 MB / 15,541 s | experimental usa menos VRAM; tarda cerca del doble |
| 12 | ~12,9 GB estimados; OOM | 1,77 GB / 65,885 s | base falla; experimental completa |

Son datos anteriores documentados en el proyecto, no resultados reproducidos aquí. El mismo documento advierte que `SPLIT>1` no está validado como correcto con el número de pasadas predeterminado.

## Estado de cobertura

| Grupo | Estado |
|---|---|
| CPU single, ambos modos | Corrido hasta el límite temporal; no se alcanzó OOM. |
| CPU multi adaptativo | Corrido hasta el límite temporal; no se alcanzó OOM. |
| CPU multi uniforme | Alcanzó el fallo de asignación en nivel 10 bajo el límite controlado de 5 GiB. |
| external/MixedOcTree, superficie | Llegó al nivel impreso 8; `std::bad_alloc` al comenzar el refinamiento siguiente, con RSS cercano a 5 GiB. |
| external/MixedOcTree, dominio completo | Llegó al nivel impreso 6; `std::bad_alloc` al comenzar el refinamiento siguiente, con RSS cercano a 5 GiB. |
| GPU base | Compila; ejecución bloqueada por el driver del entorno. |
| GPU experimental | Compila; ejecución bloqueada por el driver del entorno. |
| GPU experimental2 y test de unión | Compila; ejecución del test bloqueada por el driver del entorno. |
| Verificación CPU single/multi | VTK idénticos en nivel 5, ambos modos; discrepancia adaptativa frente a la referencia en niveles 7 y 8. |

Para completar las corridas GPU hace falta exponer una GPU NVIDIA compatible y un driver accesible por WSL2. Para alcanzar OOM en CPU single habría que dejar correr mucho más: las pruebas actuales quedaron consumidas por el tiempo de clasificación/balance antes del límite de memoria.
