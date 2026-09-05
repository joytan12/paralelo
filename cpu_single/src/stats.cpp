#include "stats.h"

#include <chrono>
#include <cstdio>
#include <cstring>

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
// Pico de memoria residente del proceso.
//
// VmHWM ("high water mark") es el máximo de RSS alcanzado, expresado en kB
// por el kernel. Fuera de Linux el archivo no existe y se devuelve 0.
// --------------------------------------------------------------------------
size_t ramPeakRSS() {
    FILE* fp = fopen("/proc/self/status", "r");
    if (!fp) return 0;

    char line[256];
    size_t peak_kb = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%zu", &peak_kb);
            break;
        }
    }
    fclose(fp);
    return peak_kb * 1024;
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
// "Octree" es exacto: hojas * sizeof(Cube). "Pico etapa" es el máximo
// calculado de RAM viva a la vez durante la etapa (arreglo de entrada +
// salida + auxiliares, o los dos buffers de la poda). Al pie se contrasta
// con el pico real de residencia que reporta el kernel.
// --------------------------------------------------------------------------
void printLevelTable(const LevelStat* stats, int n_stats,
                     size_t bytes_mesh, size_t bytes_cube,
                     double total_time) {
    char b1[32], b2[32];

    printf("\n=== Tiempo y RAM por nivel ===\n");
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
    printf("  Tamano de un cubo en RAM      : %s\n",
           formatBytes(bytes_cube, b1, sizeof(b1)));
    printf("  Malla en RAM (constante)      : %s\n",
           formatBytes(bytes_mesh, b1, sizeof(b1)));
    printf("  Pico calculado del algoritmo  : %s\n",
           formatBytes(peak_calc, b1, sizeof(b1)));

    const size_t rss = ramPeakRSS();
    if (rss > 0) {
        printf("  Pico real del proceso (VmHWM) : %s\n",
               formatBytes(rss, b1, sizeof(b1)));
    }
    printf("  Tiempo total de refinamiento  : %.3f s\n", total_time);
}
