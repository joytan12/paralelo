#include "stats.cuh"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// --------------------------------------------------------------------------
// Estado interno de la medición de VRAM.
// --------------------------------------------------------------------------
static size_t g_baseline   = 0;  // VRAM ocupada antes de reservar nada nuestro
static size_t g_stage_peak = 0;
static size_t g_run_peak   = 0;

// Lectura cruda del driver: bytes ocupados en el dispositivo.
static size_t deviceUsedRaw() {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return 0;
    return total_bytes - free_bytes;
}

void gpuSetBaseline() {
    g_baseline   = deviceUsedRaw();
    g_stage_peak = 0;
    g_run_peak   = 0;
}

size_t gpuVramUsed() {
    const size_t used = deviceUsedRaw();
    // Otro proceso puede liberar memoria y dejar la lectura bajo el baseline.
    return used > g_baseline ? used - g_baseline : 0;
}

size_t gpuVramTotal() {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return 0;
    return total_bytes;
}

size_t gpuVramBaseline() { return g_baseline; }

void gpuResetStagePeak() { g_stage_peak = 0; }

void gpuSampleVram() {
    const size_t used = gpuVramUsed();
    if (used > g_stage_peak) g_stage_peak = used;
    if (used > g_run_peak)   g_run_peak   = used;
}

size_t gpuStagePeak() { return g_stage_peak; }
size_t gpuRunPeak()   { return g_run_peak; }

// --------------------------------------------------------------------------
// Guardián de VRAM física: ver justificación en stats.cuh.
//
// "total_bytes" de cudaMemGetInfo es la VRAM dedicada real del dispositivo;
// no cambia si el driver decide desbordar a RAM del sistema. Por eso basta
// comparar contra ese número para saber, ANTES de intentar la reserva, si
// hace falta más que la VRAM física — sin depender de que cudaMalloc falle
// (con WDDM puede no fallar nunca, solo volverse muy lento).
// --------------------------------------------------------------------------
void gpuRequireBudget(size_t additional_bytes, const char* what) {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;

    const size_t used = total_bytes - free_bytes;
    if (used + additional_bytes <= total_bytes) return;

    char b_need[32], b_free[32], b_total[32];
    fprintf(stderr,
            "Error: %s necesita %s adicionales, pero solo hay %s libres de "
            "%s de VRAM fisica.\n"
            "Se detiene aqui: continuar dejaria que el driver desborde a RAM "
            "del sistema en silencio, y el proceso debe vivir integro en "
            "VRAM.\n",
            what,
            formatBytes(additional_bytes, b_need, sizeof(b_need)),
            formatBytes(free_bytes, b_free, sizeof(b_free)),
            formatBytes(total_bytes, b_total, sizeof(b_total)));
    exit(EXIT_FAILURE);
}

// --------------------------------------------------------------------------
// Reloj de pared.
// --------------------------------------------------------------------------
double wallTime() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point origin = clock::now();
    const std::chrono::duration<double> delta = clock::now() - origin;
    return delta.count();
}

// --------------------------------------------------------------------------
// Formato legible de bytes.
// --------------------------------------------------------------------------
const char* formatBytes(size_t bytes, char* buf, size_t buf_size) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = (double)bytes;
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        unit++;
    }
    if (unit == 0) snprintf(buf, buf_size, "%.0f %s", value, units[unit]);
    else           snprintf(buf, buf_size, "%.2f %s", value, units[unit]);
    return buf;
}

// --------------------------------------------------------------------------
// Tabla de tiempo y espacio por nivel.
//
// La columna "Octree" es exacta: hojas * sizeof(Cube). La columna "Pico"
// es el máximo calculado de VRAM viva simultáneamente durante la etapa
// (arreglo de entrada + salida + auxiliares, o los dos buffers de la poda).
// Al pie se contrasta con lo que reportó el driver, que redondea a los
// bloques de su suballocador y por eso es más grueso en niveles bajos.
// --------------------------------------------------------------------------
void printLevelTable(const LevelStat* stats, int n_stats,
                     size_t bytes_mesh, size_t bytes_cube,
                     double total_time) {
    char b1[32], b2[32];

    printf("\n=== Tiempo y VRAM por nivel ===\n");
    printf(" Nivel |      Hojas |     Dentro |      Borde |    Octree | Pico etapa | Tiempo (s)\n");
    printf("-------+------------+------------+------------+-----------+------------+-----------\n");

    size_t peak_calc = 0;
    for (int i = 0; i < n_stats; i++) {
        const LevelStat& s = stats[i];
        if (s.bytes_peak > peak_calc) peak_calc = s.bytes_peak;

        printf(" %5d | %10d | %10d | %10d | %9s | %10s | %9.3f\n",
               s.level, s.n_leaves, s.n_inside, s.n_border,
               formatBytes(s.bytes_octree, b1, sizeof(b1)),
               formatBytes(s.bytes_peak,   b2, sizeof(b2)),
               s.seconds);
    }

    printf("-------+------------+------------+------------+-----------+------------+-----------\n");
    printf("  Tamano de un cubo en VRAM     : %s\n",
           formatBytes(bytes_cube, b1, sizeof(b1)));
    printf("  Malla en VRAM (constante)     : %s\n",
           formatBytes(bytes_mesh, b1, sizeof(b1)));
    printf("  Pico calculado del algoritmo  : %s\n",
           formatBytes(peak_calc, b1, sizeof(b1)));
    printf("  Pico medido por el driver     : %s\n",
           formatBytes(gpuRunPeak(), b1, sizeof(b1)));
    printf("  Contexto CUDA (linea base)    : %s\n",
           formatBytes(gpuVramBaseline(), b1, sizeof(b1)));
    printf("  VRAM total del dispositivo    : %s\n",
           formatBytes(gpuVramTotal(), b1, sizeof(b1)));
    printf("  Tiempo total de refinamiento  : %.3f s\n", total_time);
}
