#include "octree.cuh"
#include "geometry.cuh"
#include "mesh.h"
#include "mesh_classify.cuh"
#include "stats.cuh"
#include "vtk_writer.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// --------------------------------------------------------------------------
// Macro para verificar errores de CUDA.
// --------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error en %s:%d — %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

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

static void printUsage(const char* program) {
    printf("Uso: %s [niveles] [modelo.mdl] [opciones]\n\n", program);
    printf("Argumentos posicionales:\n");
    printf("  niveles         Profundidad máxima del octree (1 a 20). Por defecto 3.\n");
    printf("  modelo.mdl      Malla triangular de entrada. Por defecto cortex.mdl.\n\n");
    printf("Opciones:\n");
    printf("  -q, --quiet     Omite los prints intermedios por etapa.\n");
    printf("                  Conserva el resumen y la tabla de tiempo/VRAM.\n");
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
//   Flujo por etapa: refinar -> clasificar -> podar.
//
//   El refinamiento es adaptativo con balance 2:1 por defecto, o uniforme
//   con --uniform. En ambos modos la malla y los cubos permanecen en VRAM
//   durante todo el proceso; al host solo regresan contadores enteros y el
//   resultado final que se exporta a VTK.
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

    printf("=== Octree CUDA — Refinamiento con Malla ===\n");
    printf("Niveles de refinamiento: %d\n", opt.max_level);
    printf("Modo de refinamiento   : %s\n", mode_name);
    printf("Modelo: %s\n\n", opt.mdl_path);

    // ---- Cargar el modelo .mdl ----
    Mesh mesh;
    if (!loadMDL(opt.mdl_path, mesh)) {
        fprintf(stderr, "Error cargando el modelo '%s'\n", opt.mdl_path);
        return EXIT_FAILURE;
    }

    // VRAM constante que ocupará la malla. Se calcula antes de liberar la
    // copia de RAM, porque freeMesh() pone los contadores en cero.
    const size_t bytes_mesh = (size_t)mesh.n_verts * sizeof(float3) +
                              (size_t)mesh.n_tris  * sizeof(int3);

    // Forzar la creación del contexto CUDA y fijar la línea base de VRAM,
    // para que las mediciones por nivel reflejen solo lo que reserva el
    // algoritmo y no el contexto del driver.
    CUDA_CHECK(cudaFree(0));
    gpuSetBaseline();

    // ---- Subir vértices y triángulos al device ----
    float3* d_verts = nullptr;
    int3*   d_tris  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_verts, mesh.n_verts * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_verts, mesh.vertices,
                           mesh.n_verts * sizeof(float3),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_tris, mesh.n_tris * sizeof(int3)));
    CUDA_CHECK(cudaMemcpy(d_tris, mesh.triangles,
                           mesh.n_tris * sizeof(int3),
                           cudaMemcpyHostToDevice));
    const int n_tris = mesh.n_tris;

    // ---- Cubo raíz envolvente (cúbico) ----
    Cube root;
    root.center    = mesh.center;
    root.half_size = mesh.half_size;
    root.level     = 0;
    root.state     = STATE_UNCLASSIFIED;
    root.grid_index = make_int3(0, 0, 0);
    root.path_code = 1ULL;

    printf("Cubo raíz:\n");
    printf("  Centro    : (%.2f, %.2f, %.2f)\n",
           root.center.x, root.center.y, root.center.z);
    printf("  Half-size : %.2f\n\n", root.half_size);

    // Copiar el cubo raíz al device
    Cube* d_current = nullptr;
    CUDA_CHECK(cudaMalloc(&d_current, sizeof(Cube)));
    CUDA_CHECK(cudaMemcpy(d_current, &root, sizeof(Cube), cudaMemcpyHostToDevice));
    int n_current = 1;

    // Desde este punto la geometría de entrada ya está en VRAM. Liberar la
    // copia de vértices/triángulos en RAM evita mantener dos copias durante
    // todo el refinamiento.
    freeMesh(mesh);

    // Clasificar la raíz antes de decidir si debe subdividirse.
    launchClassificationMesh(d_current, n_current, d_verts, d_tris, n_tris);
    gpuSampleVram();

    // ---- Registro de tiempo y espacio por nivel ----
    LevelStat stats[21];
    int       n_stats = 0;

    const double t_start = wallTime();

    // ---- Bucle: refinar -> clasificar -> podar ----
    for (int pass = 1; pass <= opt.max_level && n_current > 0; pass++) {
        logStage("--- Etapa %d ---\n", pass);

        gpuResetStagePeak();
        const double t_stage  = wallTime();
        const int    n_before = n_current;

        // 1) Refinar según el modo seleccionado.
        Cube* d_refined = nullptr;
        int   n_refined = 0;
        int   n_refined_parents = 0;

        if (opt.uniform) {
            // Refinamiento uniforme: todas las hojas vivas se subdividen.
            launchRefinement(d_current, n_current, &d_refined, &n_refined);
            n_refined_parents = n_current;
        } else {
            launchAdaptiveRefinement(d_current, n_current, opt.max_level,
                                      &d_refined, &n_refined,
                                      &n_refined_parents);
        }

        if (n_refined_parents == 0) {
            if (d_refined) CUDA_CHECK(cudaFree(d_refined));
            logStage("  No quedan cubos BORDER por refinar.\n");
            break;
        }

        CUDA_CHECK(cudaFree(d_current));

        logStage("  Padres refinados: %d | Hojas: %d -> %d\n",
                 n_refined_parents, n_current, n_refined);

        // 2) Clasificar contra la malla
        launchClassificationMesh(d_refined, n_refined,
                                  d_verts, d_tris, n_tris);

        // La clasificación y los cubos permanecen en device. No se copia el
        // arreglo de cubos a la RAM en las etapas intermedias.
        logStage("  Clasificación: ejecutada en GPU\n");

        // 3) Podar: eliminar los cubos "fuera"
        Cube* d_pruned = nullptr;
        int   n_pruned  = 0;
        launchPrune(d_refined, n_refined, &d_pruned, &n_pruned);
        CUDA_CHECK(cudaFree(d_refined));

        logStage("  Poda: %d -> %d hojas  (eliminados: %d)\n",
                 n_refined, n_pruned, n_refined - n_pruned);

        d_current = d_pruned;
        n_current = n_pruned;

        // 4) Registrar tiempo y espacio de la etapa. Los conteos por estado
        //    se calculan en device: solo regresan los enteros.
        LevelStat& row   = stats[n_stats++];
        row.level        = pass;
        row.n_leaves     = n_current;
        row.n_inside     = launchCountByState(d_current, n_current, STATE_INSIDE);
        row.n_border     = launchCountByState(d_current, n_current, STATE_BORDER);
        row.bytes_octree = (size_t)n_current * sizeof(Cube);
        row.seconds      = wallTime() - t_stage;

        // Pico exacto de VRAM viva durante la etapa. Se comparan los dos
        // momentos de mayor ocupacion y se toma el mayor:
        //   refinar: entrada + salida + auxiliares
        //   podar  : salida del refinamiento + buffer de poda (peor caso)
        const size_t scratch = opt.uniform
                             ? 0
                             : (size_t)n_before * adaptiveScratchBytesPerLeaf();
        const size_t peak_refine = (size_t)n_before  * sizeof(Cube)
                                 + (size_t)n_refined * sizeof(Cube) + scratch;
        const size_t peak_prune  = 2 * (size_t)n_refined * sizeof(Cube);
        row.bytes_peak = bytes_mesh +
                         (peak_refine > peak_prune ? peak_refine : peak_prune);

        logStage("  VRAM etapa: pico %.2f MB | tiempo %.3f s\n\n",
                 row.bytes_peak / (1024.0 * 1024.0), row.seconds);
    }

    const double total_time = wallTime() - t_start;

    // ---- Copiar resultado final al host ----
    printf("\n=== Resultado final: %d cubos sobrevivientes ===\n", n_current);

    Cube* h_cubes = (Cube*)malloc((size_t)n_current * sizeof(Cube));
    if (!h_cubes) {
        fprintf(stderr, "Error: malloc falló para %d cubos.\n", n_current);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_cubes, d_current, (size_t)n_current * sizeof(Cube),
                           cudaMemcpyDeviceToHost));

    // Estadísticas finales
    int n_inside = 0, n_border = 0;
    for (int i = 0; i < n_current; i++) {
        if (h_cubes[i].state == STATE_INSIDE) n_inside++;
        if (h_cubes[i].state == STATE_BORDER) n_border++;
    }
    printf("  Dentro: %d | Borde: %d\n", n_inside, n_border);

    // Comparar con un refinamiento uniforme de profundidad max_level
    long long uniforme = (long long)round(pow(8.0, opt.max_level));
    printf("  Refinamiento uniforme produciría: %lld cubos (reducción: %.1f%%)\n",
           uniforme, 100.0 * (1.0 - (double)n_current / uniforme));

    // ---- Tabla de tiempo y VRAM por nivel ----
    printLevelTable(stats, n_stats, bytes_mesh, sizeof(Cube), total_time);

    // ---- Exportar a VTK ----
    char vtk_filename[256];
    snprintf(vtk_filename, sizeof(vtk_filename),
             "output/octree_%s_level_%d.vtk",
             opt.uniform ? "uniform" : "pruned", opt.max_level);
    writeVTK(vtk_filename, h_cubes, n_current);

    // ---- Limpieza ----
    free(h_cubes);
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_verts));
    CUDA_CHECK(cudaFree(d_tris));

    printf("\n¡Refinamiento completado exitosamente!\n");
    return EXIT_SUCCESS;
}
