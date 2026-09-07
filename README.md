# PROYECTO OCTREE — REFINAMIENTO Y CLASIFICACIÓN DE MALLAS 3D

Este proyecto implementa un algoritmo de **refinamiento y clasificación de Octrees en 3D** a partir de una malla de triángulos (modelo tridimensional en formato `.mdl`). 

En la versión CUDA, la malla triangular y los cubos permanecen en memoria de
la GPU durante el refinamiento. La clasificación, el balance topológico 2:1,
la generación de hijos y la poda se ejecutan en device; la RAM se utiliza para
leer inicialmente el `.mdl` y para recibir el resultado final que se exporta a
VTK.

El algoritmo clasifica cada cubo del octree en tres estados respecto a la malla:
*   **INSIDE (Dentro):** El cubo está completamente contenido dentro de la malla.
*   **OUTSIDE (Fuera):** El cubo está completamente fuera de la malla (este se poda/elimina).
*   **BORDER (Borde):** El cubo intersecta la superficie de la malla (este se subdivide en 8 hijos en el siguiente nivel).

Para la entrega, el proyecto consta de **tres versiones** altamente optimizadas:
1.  **CUDA (GPU)** — Ubicada en la raíz del proyecto.
2.  **CPU Secuencial (1 hilo)** — Ubicada en la carpeta `cpu_single/`.
3.  **CPU Multihilo (OpenMP)** — Ubicada en la carpeta `cpu_multi/`.

---

## 🔀 MODOS DE REFINAMIENTO

Las tres versiones soportan **dos modos**, seleccionables con la variable `MODE`:

### `MODE=adaptive` — Adaptativo con balance 2:1 (por defecto)

Los cubos `INSIDE` se conservan como hojas y solo se subdividen los cubos
`BORDER`. Después de cada etapa se cierra un balance topológico **2:1**: dos
hojas que se tocan por cara, arista o vértice no pueden diferir en más de un
nivel. Para lograrlo también pueden subdividirse vecinos `INSIDE` más gruesos.
`LEVEL` es la profundidad máxima; no obliga a refinar todo el dominio.

### `MODE=uniform` — Uniforme

Todas las hojas vivas se subdividen en cada nivel, sin la regla de balance.
El resultado tiene la misma resolución en la superficie que el modo adaptativo,
pero conserva a máxima profundidad también el interior del modelo, por lo que
produce bastantes más cubos.

En ambos modos se poda igual: los cubos `OUTSIDE` se eliminan en cada etapa.

**Comparación a nivel 5 con `cortex.mdl`** (idéntica en las tres versiones):

| Modo | Hojas finales | Dentro | Borde |
| :--- | ---: | ---: | ---: |
| `adaptive` | 5.501 | 2.029 | 3.472 |
| `uniform`  | 10.646 | 7.174 | 3.472 |

El mismo detalle de superficie (3.472 cubos borde) con **la mitad de las hojas**:
esa es la ganancia del modo adaptativo.

---

## 📋 REQUISITOS DEL SISTEMA

Para compilar y ejecutar las distintas versiones del proyecto, asegúrate de contar con los siguientes elementos instalados en tu sistema:

### 1. Versión CUDA (GPU)
*   **NVIDIA CUDA Toolkit** (`nvcc` instalado y en el PATH).
*   **GPU NVIDIA compatible**. Por defecto, el Makefile está configurado para la arquitectura `sm_89` (Ada Lovelace, por ejemplo, RTX 4050/4060/4070/4080/4090 o similares). 
    > [!TIP]
    > Si tu GPU pertenece a otra arquitectura (ej. Ampere `sm_86`, Turing `sm_75`), puedes compilar indicando el parámetro `ARCH=sm_XX` en la línea de comando.

### 2. Versiones CPU (1 Hilo & Multihilo)
*   **GCC / G++** con soporte para el estándar **C++17** (versión 7 o superior).
*   **OpenMP** (soportado nativamente por la mayoría de las instalaciones de GCC/G++ en Linux).

### 3. Visualización
*   **ParaView** (opcional, para abrir y renderizar las salidas geométricas `.vtk`). Descárgalo desde [paraview.org](https://www.paraview.org/).

### Comandos de Verificación en Linux
Puedes comprobar tus herramientas ejecutando:
```bash
# Verificar compilador de CUDA
nvcc --version

# Verificar compilador de C++
g++ --version

# Verificar soporte de OpenMP
echo | g++ -fopenmp -x c - -o /dev/null && echo "OpenMP: OK"
```

---

## 📂 ESTRUCTURA DEL PROYECTO

```text
paralelo/
├── Makefile              # Makefile para la versión CUDA (GPU)
├── README.md             # Esta guía de usuario
├── INSTRUCTIVO.txt       # Instructivo rápido de comandos
├── verificar.sh          # Comprobación cruzada de las tres versiones
├── cortex.mdl            # Modelo 3D de entrada (por defecto)
├── src/                  # Código fuente de la versión CUDA (GPU) y VTK
├── output/               # Salidas VTK de la versión CUDA (se crea automáticamente)
├── vtk_refinement_analyzer/ # Analizador C++ de niveles y balance 2:1
│
├── cpu_single/           # Versión CPU Secuencial (1 hilo)
│   ├── Makefile          # Makefile de la versión secuencial
│   ├── src/              # Código fuente C++ secuencial
│   └── output/           # Salidas VTK (se crea automáticamente)
│
└── cpu_multi/            # Versión CPU Paralela (OpenMP)
    ├── Makefile          # Makefile de la versión paralela
    ├── src/              # Código fuente C++ OpenMP
    └── output/           # Salidas VTK (se crea automáticamente)
```

---

## 🚀 INSTRUCCIONES DE COMPILACIÓN Y EJECUCIÓN

Todas las versiones se pueden compilar y ejecutar de forma independiente. A continuación se detallan los comandos para cada una:

### 1. Versión CUDA (GPU)
Ejecuta los siguientes comandos desde la **raíz del proyecto** (`paralelo/`):

*   **Compilar:**
    ```bash
    make
    ```
    *(Opcional: Si tienes otra GPU, ej. RTX 3060, puedes usar `make ARCH=sm_86`)*

*   **Ejecutar con parámetros por defecto** (Refinamiento nivel 3, modelo `cortex.mdl`):
    ```bash
    make run
    ```

*   **Ejecutar con parámetros personalizados:**
    ```bash
    make run LEVEL=5
    make run LEVEL=5 QUIET=1                 # sin prints intermedios
    make run LEVEL=5 MODE=uniform            # refina todas las hojas
    make run LEVEL=4 MODE=uniform QUIET=1 MDL=cortex.mdl
    ```

*   **Ver todas las opciones:**
    ```bash
    make help
    ./octree --help
    ```

*   **Limpiar binarios y objetos:**
    ```bash
    make clean
    ```

---

### 2. Versión CPU - Secuencial (cpu_single/)
Puedes compilar y ejecutar esta versión directamente desde la raíz del proyecto usando la bandera `-C`:

*   **Compilar:**
    ```bash
    make -C cpu_single
    ```

*   **Ejecutar con parámetros por defecto:**
    ```bash
    make -C cpu_single run
    ```

*   **Ejecutar con parámetros personalizados:**
    ```bash
    make -C cpu_single run LEVEL=5 MDL=../cortex.mdl
    make -C cpu_single run LEVEL=5 QUIET=1
    make -C cpu_single run LEVEL=5 MODE=uniform
    ```

*   **Limpiar:**
    ```bash
    make -C cpu_single clean
    ```

*(Alternativamente, puedes entrar a la carpeta `cd cpu_single`, y ejecutar `make`, `make run LEVEL=4` o directamente `./octree_cpu 4 ../cortex.mdl --quiet`)*.

---

### 4. Analizar un VTK generado

El analizador lee los VTK binarios de `output/` y muestra la distribución de
celdas por nivel, si el refinamiento es completo/uniforme o por niveles/
adaptativo, y si los vecinos cumplen la regla 2:1 por cara, arista o vértice.

```bash
make analyze archive=output/octree_pruned_level_5.vtk
make analyze archive=output/octree_uniform_level_5.vtk
make run analyze archive=output/octree_pruned_level_5.vtk LEVEL=5 QUIET=1
```

Devuelve código `0` si cumple 2:1 y código `2` si encuentra violaciones.
También puede compilarse o ejecutarse directamente:

```bash
make -C vtk_refinement_analyzer
make -C vtk_refinement_analyzer run ARCHIVE=../output/archivo.vtk
```

---

### 3. Versión CPU - Multihilo OpenMP (cpu_multi/)
Puedes gestionar esta versión paralela desde la raíz del proyecto:

*   **Compilar:**
    ```bash
    make -C cpu_multi
    ```

*   **Ejecutar usando todos los hilos del CPU** (por defecto `nproc`):
    ```bash
    make -C cpu_multi run
    ```

*   **Ejecutar con parámetros personalizados:**
    ```bash
    make -C cpu_multi run THREADS=4
    make -C cpu_multi run LEVEL=5 THREADS=8
    make -C cpu_multi run LEVEL=5 THREADS=8 QUIET=1
    make -C cpu_multi run LEVEL=5 MODE=uniform THREADS=8
    ```

*   **Limpiar:**
    ```bash
    make -C cpu_multi clean
    ```

*(Alternativamente, puedes ejecutar fijando la variable de entorno OpenMP en la carpeta `cpu_multi`):*
```bash
cd cpu_multi
OMP_NUM_THREADS=4 ./octree_cpu_mt 4 ../cortex.mdl --quiet
```

---

## ⚙️ PARÁMETROS DE EJECUCIÓN

| Parámetro | Descripción | Rango / Valores sugeridos |
| :--- | :--- | :--- |
| `LEVEL` | Profundidad máxima del Octree. Cada subdivisión produce 8 hijos. | `1` a `20`. Sugeridos: `3` (rápido), `5` (moderado), `7` o superior (cómputo intensivo). |
| `MDL` | Ruta al archivo del modelo 3D triangular de entrada. | Archivos `.mdl`. Por defecto `cortex.mdl` en la raíz. |
| `MODE` | Modo de refinamiento. | `adaptive` (por defecto, balance 2:1) o `uniform` (subdivide todas las hojas). |
| `QUIET` | Omite los prints intermedios por etapa. Conserva el resumen final y la tabla de tiempo/memoria. | `0` (por defecto) o `1`. |
| `THREADS` | (Solo en CPU Multihilo) Cantidad de hilos de ejecución OpenMP a utilizar. | Por defecto usa todos los núcleos detectados (`nproc`). |
| `ARCH` | (Solo en CUDA) Arquitectura de computación de la GPU NVIDIA destino. | Por defecto `sm_89`. Ejemplos: `sm_86` (Ampere), `sm_75` (Turing), `sm_70` (Volta). |

Las mismas opciones existen como banderas del ejecutable: `--quiet` / `-q`,
`--uniform` / `-u`, `--adaptive` / `-a`, `--help` / `-h`.

---

## 📊 INDICADORES DE TIEMPO Y MEMORIA

Las tres versiones imprimen al final una tabla con el **tiempo** y el **espacio
que ocupa el modelo en cada nivel**. Esta tabla se muestra siempre, incluso con
`QUIET=1`.

Ejemplo de la versión CUDA (`make run LEVEL=5 QUIET=1`):

```text
=== Tiempo y VRAM por nivel ===
 Nivel |      Hojas |     Dentro |      Borde |    Octree | Pico etapa | Tiempo (s)
-------+------------+------------+------------+-----------+------------+-----------
     1 |          8 |          0 |          8 |     384 B |  111.52 KB |     0.004
     2 |         55 |          1 |         54 |   2.58 KB |  116.77 KB |     0.003
     3 |        252 |         50 |        202 |  11.81 KB |  151.36 KB |     0.003
     4 |       1173 |        325 |        848 |  54.98 KB |  267.61 KB |     0.003
     5 |       5501 |       2029 |       3472 | 257.86 KB |  807.42 KB |     0.004
-------+------------+------------+------------+-----------+------------+-----------
  Tamano de un cubo en VRAM     : 48 B
  Malla en VRAM (constante)     : 110.77 KB
  Pico calculado del algoritmo  : 807.42 KB
  Pico medido por el driver     : 4.00 MB
  Contexto CUDA (linea base)    : 1.04 GB
  VRAM total del dispositivo    : 6.00 GB
  Tiempo total de refinamiento  : 0.018 s
```

### Cómo leer las columnas

| Columna | Significado |
| :--- | :--- |
| `Octree` | Memoria exacta que ocupa el modelo en ese nivel: `hojas × sizeof(Cube)`. **Este es el dato de cuánta memoria ocupa el octree por nivel.** |
| `Pico etapa` | Máximo de memoria viva simultáneamente durante esa etapa. Es el mayor entre el momento del refinamiento (entrada + salida + auxiliares) y el de la poda (dos arreglos completos). Incluye la malla. |
| `Tiempo (s)` | Tiempo de pared de esa etapa: refinar + clasificar + podar. |

### Notas sobre las cifras

*   **`sizeof(Cube)` difiere entre versiones:** 48 B en CUDA y 24 B en las
    versiones CPU. La estructura de GPU carga además `grid_index` y
    `path_code`, que son los que permiten localizar vecinos en device sin
    devolver el octree a la RAM.
*   **CUDA reporta dos picos.** El *calculado* es exacto y comparable entre
    niveles. El *medido por el driver* proviene de `cudaMemGetInfo` y se
    redondea a los bloques de su suballocador (múltiplos de ~2 MB), así que en
    niveles bajos aparece inflado.
*   **La línea base del contexto CUDA** (~1 GB) es memoria del driver, no del
    algoritmo, y por eso se descuenta de las mediciones por nivel.
*   **Las versiones CPU reportan `VmHWM`**, el pico real de memoria residente
    del proceso según el kernel. Siempre es mayor que el pico calculado porque
    incluye el ejecutable, la librería estándar y el heap.

---

## ✅ COMPROBACIÓN CRUZADA

El script `verificar.sh` compila las tres versiones, las ejecuta en los dos
modos y contrasta los conteos finales, que deben coincidir exactamente:

```bash
./verificar.sh        # nivel 5 por defecto
./verificar.sh 6
```

---

## 📊 VISUALIZACIÓN DE RESULTADOS (.vtk)

Cada versión escribe un archivo de salida en formato VTK Legacy en su respectiva carpeta `output/`.

**Modo adaptativo (por defecto):**
*   **CUDA:** `output/octree_pruned_level_N.vtk`
*   **CPU Secuencial:** `cpu_single/output/octree_cpu_single_level_N.vtk`
*   **CPU Multihilo:** `cpu_multi/output/octree_cpu_multi_level_N.vtk`

**Modo uniforme (`MODE=uniform`):**
*   **CUDA:** `output/octree_uniform_level_N.vtk`
*   **CPU Secuencial:** `cpu_single/output/octree_cpu_single_uniform_level_N.vtk`
*   **CPU Multihilo:** `cpu_multi/output/octree_cpu_multi_uniform_level_N.vtk`

Los dos modos escriben archivos distintos, así que puedes generar ambos y
compararlos lado a lado en ParaView.

### Pasos para visualizar en ParaView:
1.  Abre **ParaView**.
2.  Ve a `File` -> `Open` y selecciona el archivo `.vtk` generado.
3.  Haz clic en el botón verde **Apply** en el panel de propiedades (izquierda).
4.  En la barra de herramientas de visualización, cambia la representación de `Outline` a `Surface` o `Surface With Edges`.
5.  En el menú desplegable de coloración, selecciona la propiedad **state** (0 = fuera, 1 = dentro, 2 = borde) o **level** para colorear los cubos según su profundidad y clasificación.

---

## 🧼 LIMPIEZA RÁPIDA DE TODO EL PROYECTO

Para dejar el directorio limpio antes de comprimirlo o entregarlo (eliminando ejecutables compilados, archivos intermedios de la carpeta `build/` y archivos `.vtk` temporales), ejecuta desde la raíz:

```bash
make clean && make -C cpu_single clean && make -C cpu_multi clean
```
