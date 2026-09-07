#ifndef STATS_CUH
#define STATS_CUH

#include <cstddef>

// --------------------------------------------------------------------------
// Indicadores de tiempo y espacio del refinamiento.
//
// Se registra una fila por etapa (un nivel de refinamiento) con la cantidad
// de hojas vivas, la VRAM que ocupa el octree y el pico de VRAM medido
// durante esa etapa. Al final se imprime la tabla completa.
// --------------------------------------------------------------------------
struct LevelStat {
    int    level;         // nivel alcanzado en esta etapa
    int    n_leaves;      // hojas vivas al cerrar la etapa (tras podar)
    int    n_inside;      // hojas DENTRO
    int    n_border;      // hojas BORDE
    size_t bytes_octree;  // hojas vivas * sizeof(Cube)
    size_t bytes_peak;    // pico calculado de VRAM viva durante la etapa
    double seconds;       // tiempo de pared de la etapa
};

// --------------------------------------------------------------------------
// Medición de VRAM.
//
// gpuSetBaseline() se llama una vez creado el contexto CUDA. A partir de ahí
// gpuVramUsed() descuenta ese contexto y reporta solo lo que reserva el
// algoritmo, que es el número comparable entre niveles.
// --------------------------------------------------------------------------
void   gpuSetBaseline();
size_t gpuVramUsed();      // VRAM del algoritmo (driver, menos baseline)
size_t gpuVramTotal();     // VRAM total del dispositivo
size_t gpuVramBaseline();  // contexto CUDA + otros procesos al arrancar

void   gpuResetStagePeak();  // reinicia el pico al comenzar una etapa
void   gpuSampleVram();      // muestrea ahora y actualiza los picos
size_t gpuStagePeak();       // pico de la etapa actual
size_t gpuRunPeak();         // pico de toda la ejecución

// --------------------------------------------------------------------------
// Reloj de pared en segundos (monotónico).
// --------------------------------------------------------------------------
double wallTime();

// --------------------------------------------------------------------------
// Formatea bytes como "1.38 GB" sobre buf y lo devuelve.
// --------------------------------------------------------------------------
const char* formatBytes(size_t bytes, char* buf, size_t buf_size);

// --------------------------------------------------------------------------
// Imprime la tabla de tiempo y espacio por nivel.
//
//   stats        – filas registradas, una por etapa
//   n_stats      – cantidad de filas
//   bytes_mesh   – VRAM constante que ocupa la malla (vértices + triángulos)
//   total_time   – tiempo de pared de todo el refinamiento
// --------------------------------------------------------------------------
void printLevelTable(const LevelStat* stats, int n_stats,
                      size_t bytes_mesh, size_t bytes_cube,
                      double total_time);

#endif // STATS_CUH
