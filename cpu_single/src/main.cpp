#include "octree.h"
#include "mesh_classify.h"
#include "mesh.h"
#include "stats.h"
#include "vtk_writer.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// --------------------------------------------------------------------------
// Opciones de línea de comando.
// --------------------------------------------------------------------------
struct Options {
    int         max_level;
    const char* mdl_path;
    bool        quiet;     // sin prints intermedios
    bool        uniform;   // refinar todas las hojas en todos los niveles
};

// Silencia los prints de etapa cuando se pide modo --quiet.
static bool g_quiet = false;

static void logStage(const char* fmt, ...) {
    if (g_quiet) return;
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

// --------------------------------------------------------------------------
// Estadísticas de clasificación por etapa.
// --------------------------------------------------------------------------
static void logClassificationStats(const Cube* cubes, int n) {
    if (g_quiet) return;

    int n_outside = 0, n_inside = 0, n_border = 0, n_unclass = 0;
    for (int i = 0; i < n; i++) {
        switch (cubes[i].state) {
            case STATE_OUTSIDE:      n_outside++; break;
            case STATE_INSIDE:       n_inside++;  break;
            case STATE_BORDER:       n_border++;  break;
            case STATE_UNCLASSIFIED: n_unclass++; break;
        }
    }
    printf("    Dentro: %d | Borde: %d | Fuera: %d",
           n_inside, n_border, n_outside);
    if (n_unclass > 0)
        printf(" | Sin clasificar: %d", n_unclass);
    printf("\n");
}

// Cuenta cuántos cubos tienen un estado dado.
static int countByState(const Cube* cubes, int n, int target_state) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (cubes[i].state == target_state) count++;
    }
    return count;
}

static void printUsage(const char* program) {
    printf("Uso: %s [niveles] [modelo.mdl] [opciones]\n\n", program);
    printf("Argumentos posicionales:\n");
    printf("  niveles         Profundidad máxima del octree (1 a 20). Por defecto 3.\n");
    printf("  modelo.mdl      Malla triangular de entrada. Por defecto cortex.mdl.\n\n");
    printf("Opciones:\n");
    printf("  -q, --quiet     Omite los prints intermedios por etapa.\n");
    printf("                  Conserva el resumen y la tabla de tiempo/RAM.\n");
    printf("  -u, --uniform   Refinamiento uniforme: subdivide todas las hojas\n");
    printf("                  en cada nivel, sin la regla de balance 2:1.\n");
    printf("  -a, --adaptive  Refinamiento adaptativo con balance 2:1 (por defecto):\n");
    printf("                  subdivide los BORDER y los vecinos necesarios.\n");
    printf("  -h, --help      Muestra esta ayuda.\n");
}

// Devuelve false si los argumentos son inválidos.
// want_help indica que solo se pidió la ayuda.
static bool parseOptions(int argc, char* argv[], Options& opt, bool& want_help) {
    opt.max_level = 3;
    opt.mdl_path  = "cortex.mdl";
    opt.quiet     = false;
    opt.uniform   = false;
    want_help     = false;

    int positional = 0;

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];

        if (arg[0] == '-' && arg[1] != '\0') {
            if (!strcmp(arg, "-q") || !strcmp(arg, "--quiet")) {
                opt.quiet = true;
            } else if (!strcmp(arg, "-u") || !strcmp(arg, "--uniform")) {
                opt.uniform = true;
            } else if (!strcmp(arg, "-a") || !strcmp(arg, "--adaptive")) {
                opt.uniform = false;
            } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
                want_help = true;
                return true;
            } else {
                fprintf(stderr, "Opción desconocida: %s\n\n", arg);
                return false;
            }
            continue;
        }

        // Argumentos posicionales: primero el nivel, luego el modelo.
        if (positional == 0) {
            opt.max_level = atoi(arg);
            if (opt.max_level < 1 || opt.max_level > 20) {
                fprintf(stderr, "Niveles fuera de rango: %s (válido: 1 a 20)\n\n", arg);
                return false;
            }
        } else if (positional == 1) {
            opt.mdl_path = arg;
        } else {
            fprintf(stderr, "Argumento sobrante: %s\n\n", arg);
            return false;
        }
        positional++;
    }
    return true;
}

// --------------------------------------------------------------------------
// main:
//   Flujo por etapa: refinar -> clasificar -> podar.  (1 solo hilo CPU)
// --------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Options opt;
    bool want_help = false;

    if (!parseOptions(argc, argv, opt, want_help)) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (want_help) {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }
    g_quiet = opt.quiet;

    const char* mode_name = opt.uniform ? "uniforme (todas las hojas)"
                                        : "adaptativo con balance 2:1";

    printf("=== Octree CPU (1 hilo) — Refinamiento con Malla ===\n");
    printf("Niveles de refinamiento: %d\n", opt.max_level);
    printf("Modo de refinamiento   : %s\n", mode_name);
    printf("Modelo: %s\n\n", opt.mdl_path);

    // ---- Cargar el modelo .mdl ----
    Mesh mesh;
    if (!loadMDL(opt.mdl_path, mesh)) {
        fprintf(stderr, "Error cargando el modelo '%s'\n", opt.mdl_path);
        return EXIT_FAILURE;
    }

    const size_t bytes_mesh = (size_t)mesh.n_verts * sizeof(Vec3) +
                              (size_t)mesh.n_tris  * sizeof(IVec3);

    // ---- Cubo raíz envolvente ----
    Cube root;
    root.center    = mesh.center;
    root.half_size = mesh.half_size;
    root.level     = 0;
    root.state     = STATE_UNCLASSIFIED;

    printf("Cubo raíz:\n");
    printf("  Centro    : (%.2f, %.2f, %.2f)\n",
           root.center.x, root.center.y, root.center.z);
    printf("  Half-size : %.2f\n\n", root.half_size);

    // Clasificar la raíz antes de comenzar. Así el algoritmo puede decidir
    // si debe refinarla o conservarla como hoja.
    classifyMesh(&root, 1, mesh.vertices, mesh.triangles, mesh.n_tris);

    Cube* current  = (Cube*)malloc(sizeof(Cube));
    current[0]     = root;
    int   n_current = 1;

    // ---- Registro de tiempo y espacio por nivel ----
    LevelStat stats[21];
    int       n_stats = 0;

    const double t_start = wallTime();

    // ---- Bucle: refinar -> clasificar -> podar ----
    for (int pass = 1; pass <= opt.max_level && n_current > 0; pass++) {
        logStage("--- Etapa %d ---\n", pass);

        const double t_stage  = wallTime();
        const int    n_before = n_current;

        // 1) Refinar según el modo seleccionado.
        Cube* refined  = nullptr;
        int   n_refined = 0;
        int   n_refined_parents = 0;

        if (opt.uniform) {
            // Refinamiento uniforme: todas las hojas vivas se subdividen.
            refine(current, n_current, &refined, &n_refined);
            n_refined_parents = n_current;
        } else {
            refineAdaptive(current, n_current, opt.max_level,
                           &refined, &n_refined, &n_refined_parents);
        }

        if (n_refined_parents == 0) {
            free(refined);
            logStage("  No quedan cubos BORDER por refinar.\n");
            break;
        }

        free(current);

        logStage("  Padres refinados: %d | Hojas: %d -> %d\n",
                 n_refined_parents, n_current, n_refined);

        // 2) Clasificar contra la malla
        classifyMesh(refined, n_refined, mesh.vertices, mesh.triangles, mesh.n_tris);

        logStage("  Clasificación: ");
        logClassificationStats(refined, n_refined);

        // 3) Podar: eliminar los cubos "fuera"
        Cube* pruned  = nullptr;
        int   n_pruned = 0;
        prune(refined, n_refined, &pruned, &n_pruned);
        free(refined);

        logStage("  Poda: %d -> %d hojas  (eliminados: %d)\n",
                 n_refined, n_pruned, n_refined - n_pruned);

        current  = pruned;
        n_current = n_pruned;

        // 4) Registrar tiempo y espacio de la etapa.
        LevelStat& row   = stats[n_stats++];
        row.level        = pass;
        row.n_leaves     = n_current;
        row.n_inside     = countByState(current, n_current, STATE_INSIDE);
        row.n_border     = countByState(current, n_current, STATE_BORDER);
        row.bytes_octree = (size_t)n_current * sizeof(Cube);
        row.seconds      = wallTime() - t_stage;

        // Pico exacto de RAM viva durante la etapa. Se comparan los dos
        // momentos de mayor ocupación y se toma el mayor:
        //   refinar: entrada + salida + auxiliares
        //   podar  : salida del refinamiento + buffer de poda (peor caso,
        //            la versión secuencial reserva n_refined cubos)
        const size_t scratch = opt.uniform
                             ? 0
                             : (size_t)n_before * refineScratchBytesPerLeaf();
        const size_t peak_refine = (size_t)n_before  * sizeof(Cube)
                                 + (size_t)n_refined * sizeof(Cube) + scratch;
        const size_t peak_prune  = 2 * (size_t)n_refined * sizeof(Cube)
                                 + (size_t)n_refined * pruneScratchBytesPerLeaf();
        row.bytes_peak = bytes_mesh +
                         (peak_refine > peak_prune ? peak_refine : peak_prune);

        logStage("  RAM etapa: pico %.2f MB | tiempo %.3f s\n\n",
                 row.bytes_peak / (1024.0 * 1024.0), row.seconds);
    }

    const double total_time = wallTime() - t_start;

    // ---- Resultado final ----
    printf("\n=== Resultado final: %d cubos sobrevivientes ===\n", n_current);

    int n_inside = 0, n_border = 0;
    for (int i = 0; i < n_current; i++) {
        if (current[i].state == STATE_INSIDE) n_inside++;
        if (current[i].state == STATE_BORDER) n_border++;
    }
    printf("  Dentro: %d | Borde: %d\n", n_inside, n_border);

    long long uniforme = (long long)round(pow(8.0, opt.max_level));
    printf("  Refinamiento uniforme produciría: %lld cubos (reducción: %.1f%%)\n",
           uniforme, 100.0 * (1.0 - (double)n_current / uniforme));

    // ---- Tabla de tiempo y RAM por nivel ----
    printLevelTable(stats, n_stats, bytes_mesh, sizeof(Cube), total_time);

    // ---- Exportar a VTK ----
    char vtk_filename[256];
    snprintf(vtk_filename, sizeof(vtk_filename),
             "output/octree_cpu_single%s_level_%d.vtk",
             opt.uniform ? "_uniform" : "", opt.max_level);
    writeVTK(vtk_filename, current, n_current);

    // ---- Limpieza ----
    free(current);
    freeMesh(mesh);

    printf("\n¡Refinamiento completado exitosamente!\n");
    return EXIT_SUCCESS;
}
