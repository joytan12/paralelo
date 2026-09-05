#ifndef STATS_H
#define STATS_H

#include <cstddef>

// --------------------------------------------------------------------------
// Indicadores de tiempo y espacio del refinamiento.
//
// Se registra una fila por etapa (un nivel) con la cantidad de hojas vivas,
// la RAM que ocupa el octree y el pico de RAM de esa etapa. Al final se
// imprime la tabla completa.
// --------------------------------------------------------------------------
struct LevelStat {
    int    level;         // nivel alcanzado en esta etapa
    int    n_leaves;      // hojas vivas al cerrar la etapa (tras podar)
    int    n_inside;      // hojas DENTRO
    int    n_border;      // hojas BORDE
    size_t bytes_octree;  // hojas vivas * sizeof(Cube)
    size_t bytes_peak;    // pico calculado de RAM viva durante la etapa
    double seconds;       // tiempo de pared de la etapa
};

// --------------------------------------------------------------------------
// Reloj de pared en segundos (monotónico).
// --------------------------------------------------------------------------
double wallTime();

// --------------------------------------------------------------------------
// Pico de memoria residente del proceso, leído de /proc/self/status (VmHWM).
// Incluye el ejecutable, la librería estándar y el heap, así que es mayor
// que el pico calculado del algoritmo. Devuelve 0 fuera de Linux.
// --------------------------------------------------------------------------
size_t ramPeakRSS();

// --------------------------------------------------------------------------
// Formatea bytes como "1.38 GB" sobre buf y lo devuelve.
// --------------------------------------------------------------------------
const char* formatBytes(size_t bytes, char* buf, size_t buf_size);

// --------------------------------------------------------------------------
// Imprime la tabla de tiempo y espacio por nivel.
//
//   stats       – filas registradas, una por etapa
//   n_stats     – cantidad de filas
//   bytes_mesh  – RAM constante que ocupa la malla (vértices + triángulos)
//   bytes_cube  – sizeof(Cube), para poder verificar la columna a mano
//   total_time  – tiempo de pared de todo el refinamiento
// --------------------------------------------------------------------------
void printLevelTable(const LevelStat* stats, int n_stats,
                     size_t bytes_mesh, size_t bytes_cube,
                     double total_time);

#endif // STATS_H
