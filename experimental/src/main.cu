#include "octree.cuh"
#include "geometry.cuh"
#include "mesh.h"
#include "mesh_classify.cuh"
#include "stats.cuh"
#include "vtk_stream.cuh"
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
            fprintf(stderr, "CUDA error en %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// --------------------------------------------------------------------------
// Opciones de linea de comando.
// --------------------------------------------------------------------------
struct Options {
    int         max_level;
    const char* mdl_path;
    bool        quiet;
    bool        uniform;
    int         split_level;   // profundidad de la descomposicion (8^D subarboles)
    int         max_passes;    // pasadas externas para cerrar el balance 2:1
    bool        no_vtk;
    int         chunk_cubes;   // cubos por trozo de evacuacion
};

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
    printf("  niveles         Profundidad maxima del octree (1 a 20). Por defecto 3.\n");
    printf("  modelo.mdl      Malla triangular de entrada. Por defecto cortex.mdl.\n\n");
    printf("Opciones:\n");
    printf("  -q, --quiet     Omite los prints intermedios por etapa.\n");
    printf("  -u, --uniform   Refinamiento uniforme: subdivide todas las hojas.\n");
    printf("  -a, --adaptive  Refinamiento adaptativo con balance 2:1 (por defecto).\n");
    printf("  -s, --split-level N\n");
    printf("                  Nivel de descomposicion: se procesan 8^N subarboles\n");
    printf("                  de forma secuencial en la GPU. Por defecto 1 (8 hijos).\n");
    printf("  -p, --passes N  Maximo de pasadas externas para cerrar el balance 2:1\n");
    printf("                  entre subarboles. Por defecto 3. Con 1 se acepta el\n");
    printf("                  resultado de la primera pasada aunque no converja.\n");
    printf("      --chunk N   Cubos por trozo de evacuacion a disco. Por defecto 262144.\n");
    printf("      --no-vtk    No escribe el archivo VTK (util en niveles altos).\n");
    printf("  -h, --help      Muestra esta ayuda.\n");
}

static bool parseOptions(int argc, char* argv[], Options& opt, bool& want_help) {
    opt.max_level   = 3;
    opt.mdl_path    = "cortex.mdl";
    opt.quiet       = false;
    opt.uniform     = false;
    opt.split_level = 1;
    opt.max_passes  = 3;
    opt.no_vtk      = false;
    opt.chunk_cubes = 262144;
    want_help       = false;

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
            } else if (!strcmp(arg, "--no-vtk")) {
                opt.no_vtk = true;
            } else if (!strcmp(arg, "-s") || !strcmp(arg, "--split-level")) {
                if (++i >= argc) { fprintf(stderr, "Falta el valor de %s\n\n", arg); return false; }
                opt.split_level = atoi(argv[i]);
                if (opt.split_level < 1 || opt.split_level > 5) {
                    fprintf(stderr, "split-level fuera de rango: %s (valido: 1 a 5)\n\n", argv[i]);
                    return false;
                }
            } else if (!strcmp(arg, "-p") || !strcmp(arg, "--passes")) {
                if (++i >= argc) { fprintf(stderr, "Falta el valor de %s\n\n", arg); return false; }
                opt.max_passes = atoi(argv[i]);
                if (opt.max_passes < 1 || opt.max_passes > 10) {
                    fprintf(stderr, "passes fuera de rango: %s (valido: 1 a 10)\n\n", argv[i]);
                    return false;
                }
            } else if (!strcmp(arg, "--chunk")) {
                if (++i >= argc) { fprintf(stderr, "Falta el valor de %s\n\n", arg); return false; }
                opt.chunk_cubes = atoi(argv[i]);
                if (opt.chunk_cubes < 4096) opt.chunk_cubes = 4096;
            } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
                want_help = true;
                return true;
            } else {
                fprintf(stderr, "Opcion desconocida: %s\n\n", arg);
                return false;
            }
            continue;
        }

        if (positional == 0) {
            opt.max_level = atoi(arg);
            if (opt.max_level < 1 || opt.max_level > 20) {
                fprintf(stderr, "Niveles fuera de rango: %s (valido: 1 a 20)\n\n", arg);
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
// Tabla de cascaras: una por subarbol y por paso de refinamiento.
//
// shell[s][L] es la cascara del subarbol s en el paso que produce hojas de
// nivel L, con la bandera de subdivision que se decidio para cada una de sus
// celdas. Es lo que los demas subarboles usan como halo en ese mismo paso, de
// modo que todos ven el mismo estado del vecino.
//
// La bandera es tan importante como la celda: una hoja de frontera puede
// estar marcada por el balance interno de su subarbol, disparada por una
// celda fina que no vive en la cascara. Desde fuera esa marca no se puede
// reconstruir a partir de state y level, y sin ella la cadena de propagacion
// se corta justo en la frontera.
//
// La ranura max_level+1 guarda la cascara final, usada en la verificacion.
// --------------------------------------------------------------------------
struct ShellTable {
    int      n_subtrees;
    int      n_slots;      // max_level + 2
    Cube**   ptr;          // [s * n_slots + L]
    int**    flags;        // banderas de subdivision, mismo indexado
    int*     count;
};

static void shellTableInit(ShellTable& t, int n_subtrees, int max_level) {
    t.n_subtrees = n_subtrees;
    t.n_slots    = max_level + 2;
    const size_t n = (size_t)n_subtrees * t.n_slots;
    t.ptr   = (Cube**)calloc(n, sizeof(Cube*));
    t.flags = (int**)calloc(n, sizeof(int*));
    t.count = (int*)calloc(n, sizeof(int));
    if (!t.ptr || !t.flags || !t.count) {
        fprintf(stderr, "Error: sin memoria para la tabla de cascaras.\n");
        exit(EXIT_FAILURE);
    }
}

static void shellTableClear(ShellTable& t) {
    const size_t n = (size_t)t.n_subtrees * t.n_slots;
    for (size_t i = 0; i < n; i++) {
        if (t.ptr[i])   { CUDA_CHECK(cudaFree(t.ptr[i]));   t.ptr[i]   = nullptr; }
        if (t.flags[i]) { CUDA_CHECK(cudaFree(t.flags[i])); t.flags[i] = nullptr; }
        t.count[i] = 0;
    }
}

static void shellTableFree(ShellTable& t) {
    shellTableClear(t);
    free(t.ptr);
    free(t.flags);
    free(t.count);
    t.ptr   = nullptr;
    t.flags = nullptr;
    t.count = nullptr;
}

static size_t shellTableBytes(const ShellTable& t) {
    const size_t n = (size_t)t.n_subtrees * t.n_slots;
    size_t total = 0;
    for (size_t i = 0; i < n; i++)
        total += (size_t)t.count[i] * (sizeof(Cube) + sizeof(int));
    return total;
}

static void shellTableSet(ShellTable& t, int s, int slot,
                          Cube* p, int* fl, int n) {
    const size_t i = (size_t)s * t.n_slots + slot;
    if (t.ptr[i])   CUDA_CHECK(cudaFree(t.ptr[i]));
    if (t.flags[i]) CUDA_CHECK(cudaFree(t.flags[i]));
    t.ptr[i]   = p;
    t.flags[i] = fl;
    t.count[i] = n;
}

static Cube* deviceClone(const Cube* src, int n) {
    if (!src || n <= 0) return nullptr;
    Cube* dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, (size_t)n * sizeof(Cube)));
    CUDA_CHECK(cudaMemcpy(dst, src, (size_t)n * sizeof(Cube),
                          cudaMemcpyDeviceToDevice));
    return dst;
}

static int* deviceCloneInt(const int* src, int n) {
    if (!src || n <= 0) return nullptr;
    int* dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dst, src, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    return dst;
}

// Adyacencia entre subarboles: dos cajas de nivel D se tocan solo si sus
// indices de rejilla difieren a lo mas en 1 en cada eje. Un subarbol que no
// toca al nuestro no puede aportar ningun vecino, asi que su cascara no entra
// al halo. Con D=1 los 8 octantes se tocan todos y no cambia nada; con D=2
// cada subarbol pasa de 63 vecinos potenciales a 26 como maximo, que es lo
// que hace practico bajar la granularidad para llegar mas profundo.
static bool* g_adjacent = nullptr;   // [target * n_subtrees + other]

static void buildAdjacency(const Cube* h_roots, int n_subtrees) {
    g_adjacent = (bool*)calloc((size_t)n_subtrees * n_subtrees, sizeof(bool));
    if (!g_adjacent) {
        fprintf(stderr, "Error: sin memoria para la tabla de adyacencia.\n");
        exit(EXIT_FAILURE);
    }
    for (int a = 0; a < n_subtrees; a++) {
        for (int b = 0; b < n_subtrees; b++) {
            if (a == b) continue;
            const int dx = h_roots[a].grid_index.x - h_roots[b].grid_index.x;
            const int dy = h_roots[a].grid_index.y - h_roots[b].grid_index.y;
            const int dz = h_roots[a].grid_index.z - h_roots[b].grid_index.z;
            g_adjacent[(size_t)a * n_subtrees + b] =
                (dx >= -1 && dx <= 1) && (dy >= -1 && dy <= 1) &&
                (dz >= -1 && dz <= 1);
        }
    }
}

// Construye el halo del subarbol `target` para el paso `slot`: la
// concatenacion de las cascaras de los subarboles adyacentes en ese mismo
// paso. Se excluye la propia para que la busqueda binaria no encuentre un
// duplicado y pierda la marca sobre la hoja real.
static void buildGhost(const ShellTable& t, int target, int slot,
                       Cube** d_ghost, int** d_ghost_split, int* n_ghost) {
    *d_ghost       = nullptr;
    *d_ghost_split = nullptr;
    *n_ghost       = 0;

    long long total = 0;
    for (int s = 0; s < t.n_subtrees; s++) {
        if (!g_adjacent[(size_t)target * t.n_subtrees + s]) continue;
        total += t.count[(size_t)s * t.n_slots + slot];
    }
    if (total <= 0) return;

    CUDA_CHECK(cudaMalloc(d_ghost, (size_t)total * sizeof(Cube)));
    CUDA_CHECK(cudaMalloc(d_ghost_split, (size_t)total * sizeof(int)));

    long long offset = 0;
    for (int s = 0; s < t.n_subtrees; s++) {
        if (!g_adjacent[(size_t)target * t.n_subtrees + s]) continue;
        const size_t i = (size_t)s * t.n_slots + slot;
        const int n = t.count[i];
        if (n <= 0) continue;
        CUDA_CHECK(cudaMemcpy(*d_ghost + offset, t.ptr[i],
                              (size_t)n * sizeof(Cube),
                              cudaMemcpyDeviceToDevice));
        if (t.flags[i]) {
            CUDA_CHECK(cudaMemcpy(*d_ghost_split + offset, t.flags[i],
                                  (size_t)n * sizeof(int),
                                  cudaMemcpyDeviceToDevice));
        } else {
            CUDA_CHECK(cudaMemset(*d_ghost_split + offset, 0,
                                  (size_t)n * sizeof(int)));
        }
        offset += n;
    }
    *n_ghost = (int)total;
}

// --------------------------------------------------------------------------
// main
//
//   Fase A: descenso uniforme hasta el nivel de descomposicion. Deja una
//           hoja por subarbol y garantiza que toda hoja posterior tenga un
//           dueno unico (su ancestro de ese nivel).
//   Fase B: cada subarbol se refina hasta el nivel maximo, solo, con el halo
//           de las cascaras de los demas para que la regla 2:1 se cumpla a
//           traves de la frontera. Al terminar se evacua desde VRAM a disco.
//   Fase C: verificacion 2:1 sobre la union de las cascaras finales. Si hubo
//           violaciones se repite la fase B con las cascaras recien medidas.
//
//   Ningun Cube se acumula en RAM del host: la CPU solo hace fwrite.
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

    // No tiene sentido descomponer mas profundo que el propio refinamiento.
    int split_level = opt.split_level;
    if (split_level > opt.max_level) split_level = opt.max_level;

    // En modo uniforme no hay busqueda de vecinos: los subarboles son
    // independientes por construccion y una sola pasada basta.
    const int max_passes = opt.uniform ? 1 : opt.max_passes;

    const char* mode_name = opt.uniform ? "uniforme (todas las hojas)"
                                        : "adaptativo con balance 2:1";

    printf("=== Octree CUDA (experimental) - Descomposicion por subarboles ===\n");
    printf("Niveles de refinamiento: %d\n", opt.max_level);
    printf("Modo de refinamiento   : %s\n", mode_name);
    printf("Nivel de descomposicion: %d  (hasta %d subarboles secuenciales)\n",
           split_level, 1 << (3 * split_level));
    printf("Modelo: %s\n\n", opt.mdl_path);

    // ---- Cargar el modelo .mdl ----
    Mesh mesh;
    if (!loadMDL(opt.mdl_path, mesh)) {
        fprintf(stderr, "Error cargando el modelo '%s'\n", opt.mdl_path);
        return EXIT_FAILURE;
    }

    const size_t bytes_mesh = (size_t)mesh.n_verts * sizeof(float3) +
                              (size_t)mesh.n_tris  * sizeof(int3);

    CUDA_CHECK(cudaFree(0));
    gpuSetBaseline();

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

    // ---- Cubo raiz envolvente ----
    Cube root;
    root.center     = mesh.center;
    root.half_size  = mesh.half_size;
    root.level      = 0;
    root.state      = STATE_UNCLASSIFIED;
    root.grid_index = make_int3(0, 0, 0);
    root.path_code  = 1ULL;

    printf("Cubo raiz:\n");
    printf("  Centro    : (%.2f, %.2f, %.2f)\n",
           root.center.x, root.center.y, root.center.z);
    printf("  Half-size : %.2f\n\n", root.half_size);

    Cube* d_current = nullptr;
    CUDA_CHECK(cudaMalloc(&d_current, sizeof(Cube)));
    CUDA_CHECK(cudaMemcpy(d_current, &root, sizeof(Cube), cudaMemcpyHostToDevice));
    int n_current = 1;

    freeMesh(mesh);

    launchClassificationMesh(d_current, n_current, d_verts, d_tris, n_tris);
    gpuSampleVram();

    // ---- Registro por nivel ----
    LevelStat stats[22];
    memset(stats, 0, sizeof(stats));
    for (int i = 0; i < 22; i++) stats[i].level = i;

    const double t_start = wallTime();

    // ======================================================================
    // Fase A: descenso uniforme hasta el nivel de descomposicion.
    // ======================================================================
    logStage("--- Fase A: descenso uniforme hasta el nivel %d ---\n", split_level);

    for (int level = 1; level <= split_level && n_current > 0; level++) {
        const double t_stage  = wallTime();
        const int    n_before = n_current;

        Cube* d_refined = nullptr;
        int   n_refined = 0;
        launchRefinement(d_current, n_current, &d_refined, &n_refined);
        CUDA_CHECK(cudaFree(d_current));

        launchClassificationMesh(d_refined, n_refined, d_verts, d_tris, n_tris);

        Cube* d_pruned = nullptr;
        int   n_pruned = 0;
        launchPrune(d_refined, n_refined, &d_pruned, &n_pruned);
        CUDA_CHECK(cudaFree(d_refined));

        d_current = d_pruned;
        n_current = n_pruned;

        LevelStat& row   = stats[level];
        row.n_leaves     = n_current;
        row.n_inside     = launchCountByState(d_current, n_current, STATE_INSIDE);
        row.n_border     = launchCountByState(d_current, n_current, STATE_BORDER);
        row.bytes_octree = (size_t)n_current * sizeof(Cube);
        row.bytes_peak   = bytes_mesh +
                           (size_t)(n_before + n_refined) * sizeof(Cube);
        row.seconds      = wallTime() - t_stage;

        logStage("  Nivel %d: %d -> %d hojas (poda: %d eliminados) | %.3f s\n",
                 level, n_before, n_refined, n_refined - n_pruned, row.seconds);
    }

    if (n_current <= 0) {
        fprintf(stderr, "El modelo no dejo ninguna hoja viva en el nivel %d.\n",
                split_level);
        return EXIT_FAILURE;
    }

    const int n_subtrees = n_current;
    Cube* d_roots = d_current;   // una hoja de nivel split_level por subarbol
    d_current = nullptr;

    // Las raices de los subarboles son a lo mas 8^D celdas: caben de sobra en
    // el host y solo se usan para decidir que subarboles se tocan.
    {
        Cube* h_roots = (Cube*)malloc((size_t)n_subtrees * sizeof(Cube));
        if (!h_roots) {
            fprintf(stderr, "Error: sin memoria para las raices.\n");
            return EXIT_FAILURE;
        }
        CUDA_CHECK(cudaMemcpy(h_roots, d_roots,
                              (size_t)n_subtrees * sizeof(Cube),
                              cudaMemcpyDeviceToHost));
        buildAdjacency(h_roots, n_subtrees);
        free(h_roots);
    }

    logStage("\n  Subarboles vivos tras la fase A: %d de %d posibles\n\n",
             n_subtrees, 1 << (3 * split_level));

    // ======================================================================
    // Fase B/C: refinamiento por subarbol, evacuacion y cierre del balance.
    // ======================================================================
    ShellTable shell_prev, shell_new;
    shellTableInit(shell_prev, n_subtrees, opt.max_level);
    shellTableInit(shell_new,  n_subtrees, opt.max_level);

    char vtk_filename[256];
    snprintf(vtk_filename, sizeof(vtk_filename),
             "output/octree_%s_level_%d.vtk",
             opt.uniform ? "uniform" : "pruned", opt.max_level);

    VtkStream* vtk = opt.no_vtk ? nullptr
                                : vtkStreamOpen(vtk_filename, opt.chunk_cubes);

    long long total_leaves  = 0;
    long long total_inside  = 0;
    long long total_border  = 0;
    unsigned long long violations = 0;
    int  passes_done = 0;
    bool converged   = false;

    for (int pass = 1; pass <= max_passes; pass++) {
        passes_done = pass;
        logStage("--- Pasada %d/%d ---\n", pass, max_passes);

        gpuResetStagePeak();

        // Cada pasada recomienza desde las raices de los subarboles.
        for (int i = split_level + 1; i <= opt.max_level; i++) {
            stats[i].n_leaves = 0; stats[i].n_inside = 0; stats[i].n_border = 0;
            stats[i].bytes_octree = 0; stats[i].bytes_peak = 0; stats[i].seconds = 0.0;
        }
        shellTableClear(shell_new);
        if (vtk) vtkStreamReset(vtk);

        total_leaves = total_inside = total_border = 0;

        // Cascaras finales de esta pasada, para la verificacion.
        Cube** final_shell   = (Cube**)calloc(n_subtrees, sizeof(Cube*));
        int*   n_final_shell = (int*)calloc(n_subtrees, sizeof(int));

        for (int s = 0; s < n_subtrees; s++) {
            const double t_sub = wallTime();

            Cube* d_own = deviceClone(d_roots + s, 1);
            int   n_own = 1;

            int step = split_level + 1;
            int first_fill = split_level + 1;

            for (; step <= opt.max_level; ++step) {
                const double t_stage  = wallTime();
                const int    n_before = n_own;

                Cube* d_ghost       = nullptr;
                int*  d_ghost_split = nullptr;
                int   n_ghost       = 0;

                if (!opt.uniform)
                    buildGhost(shell_prev, s, step,
                               &d_ghost, &d_ghost_split, &n_ghost);

                Cube* d_refined   = nullptr;
                int   n_refined   = 0;
                int   n_parents   = 0;
                int*  d_split_own = nullptr;

                if (opt.uniform) {
                    launchRefinement(d_own, n_own, &d_refined, &n_refined);
                    n_parents = n_own;
                } else {
                    launchAdaptiveRefinement(d_own, n_own, d_ghost, n_ghost,
                                             d_ghost_split, opt.max_level,
                                             &d_refined, &n_refined, &n_parents,
                                             &d_split_own);

                    // La cascara de este paso se registra despues del marcado y
                    // antes de reemplazar d_own: lleva las celdas de frontera
                    // junto con la decision que se tomo sobre ellas, que es lo
                    // que el vecino necesita ver cuando le toque este paso.
                    Cube* d_shell       = nullptr;
                    int*  d_shell_split = nullptr;
                    int   n_shell       = 0;
                    launchExtractShell(d_own, d_split_own, n_own, split_level,
                                       &d_shell, &d_shell_split, &n_shell);
                    shellTableSet(shell_new, s, step,
                                  d_shell, d_shell_split, n_shell);
                }

                if (d_ghost)       CUDA_CHECK(cudaFree(d_ghost));
                if (d_ghost_split) CUDA_CHECK(cudaFree(d_ghost_split));
                if (d_split_own)   CUDA_CHECK(cudaFree(d_split_own));

                if (n_parents == 0) {
                    if (d_refined) CUDA_CHECK(cudaFree(d_refined));
                    first_fill = step + 1;
                    break;
                }

                CUDA_CHECK(cudaFree(d_own));

                launchClassificationMesh(d_refined, n_refined,
                                         d_verts, d_tris, n_tris);

                Cube* d_pruned = nullptr;
                int   n_pruned = 0;
                launchPrune(d_refined, n_refined, &d_pruned, &n_pruned);
                CUDA_CHECK(cudaFree(d_refined));

                d_own = d_pruned;
                n_own = n_pruned;

                // Acumular la fila del nivel sobre todos los subarboles.
                LevelStat& row = stats[step];
                row.n_leaves     += n_own;
                row.n_inside     += launchCountByState(d_own, n_own, STATE_INSIDE);
                row.n_border     += launchCountByState(d_own, n_own, STATE_BORDER);
                row.bytes_octree += (size_t)n_own * sizeof(Cube);
                row.seconds      += wallTime() - t_stage;

                const size_t scratch = opt.uniform
                    ? 0
                    : (size_t)(n_before + n_ghost) * adaptiveScratchBytesPerLeaf();
                const size_t peak_refine = (size_t)(n_before + n_ghost + n_refined)
                                         * sizeof(Cube) + scratch;
                const size_t peak_prune  = (size_t)(n_refined + n_own) * sizeof(Cube);
                const size_t peak_sub = bytes_mesh
                                      + shellTableBytes(shell_prev)
                                      + shellTableBytes(shell_new)
                                      + (peak_refine > peak_prune ? peak_refine : peak_prune);
                if (peak_sub > row.bytes_peak) row.bytes_peak = peak_sub;

                if (n_own == 0) { first_fill = step + 1; break; }
                if (step == opt.max_level) first_fill = opt.max_level + 1;
            }

            // Si el subarbol dejo de crecer, su cascara ya no cambia: se
            // replica en las ranuras restantes para que los vecinos la vean
            // en los pasos que faltan.
            if (!opt.uniform) {
                // El subarbol ya no crece: su cascara no cambia y nada mas se
                // divide, asi que las banderas de las ranuras restantes van en
                // cero.
                Cube* d_tail       = nullptr;
                int*  d_tail_split = nullptr;
                int   n_tail       = 0;
                launchExtractShell(d_own, nullptr, n_own, split_level,
                                   &d_tail, &d_tail_split, &n_tail);
                for (int q = first_fill; q <= opt.max_level + 1; ++q)
                    shellTableSet(shell_new, s, q,
                                  deviceClone(d_tail, n_tail),
                                  deviceCloneInt(d_tail_split, n_tail), n_tail);
                if (d_tail_split) CUDA_CHECK(cudaFree(d_tail_split));
                final_shell[s]   = d_tail;
                n_final_shell[s] = n_tail;
            }

            total_leaves += n_own;
            total_inside += launchCountByState(d_own, n_own, STATE_INSIDE);
            total_border += launchCountByState(d_own, n_own, STATE_BORDER);

            // Evacuacion: de VRAM a disco sin pasar por un arreglo de host.
            if (vtk && n_own > 0) vtkStreamAppend(vtk, d_own, n_own);

            gpuSampleVram();
            if (d_own) CUDA_CHECK(cudaFree(d_own));

            logStage("  Subarbol %2d/%d: %9d hojas | cascara %8d | %7.3f s\n",
                     s + 1, n_subtrees, n_own,
                     opt.uniform ? 0 : n_final_shell[s], wallTime() - t_sub);
        }

        // ------------------------------------------------------------------
        // Verificacion 2:1 sobre la union de las cascaras finales.
        //
        // Dentro de un subarbol el balance lo garantiza el propio algoritmo;
        // lo unico que la descomposicion puede romper esta en las fronteras,
        // y ahi viven exactamente las hojas de las cascaras.
        // ------------------------------------------------------------------
        violations = 0;
        if (!opt.uniform) {
            long long n_union = 0;
            for (int s = 0; s < n_subtrees; s++) n_union += n_final_shell[s];

            if (n_union > 0) {
                Cube* d_union = nullptr;
                CUDA_CHECK(cudaMalloc(&d_union, (size_t)n_union * sizeof(Cube)));
                long long off = 0;
                for (int s = 0; s < n_subtrees; s++) {
                    if (n_final_shell[s] <= 0) continue;
                    CUDA_CHECK(cudaMemcpy(d_union + off, final_shell[s],
                                          (size_t)n_final_shell[s] * sizeof(Cube),
                                          cudaMemcpyDeviceToDevice));
                    off += n_final_shell[s];
                }
                violations = launchVerifyBalance(d_union, (int)n_union, split_level);
                CUDA_CHECK(cudaFree(d_union));
            }
        }

        for (int s = 0; s < n_subtrees; s++)
            if (final_shell[s]) CUDA_CHECK(cudaFree(final_shell[s]));
        free(final_shell);
        free(n_final_shell);

        // Las cascaras de esta pasada pasan a ser el halo de la siguiente.
        ShellTable tmp = shell_prev;
        shell_prev = shell_new;
        shell_new  = tmp;

        if (opt.uniform) {
            converged = true;
            logStage("  Modo uniforme: subarboles independientes, sin halo.\n\n");
            break;
        }

        logStage("  Verificacion 2:1 entre subarboles: %llu violaciones\n\n",
                 violations);

        if (violations == 0) { converged = true; break; }
    }

    const double total_time = wallTime() - t_start;

    // ---- Resultado ----
    printf("\n=== Resultado final: %lld cubos sobrevivientes ===\n", total_leaves);
    printf("  Dentro: %lld | Borde: %lld\n", total_inside, total_border);

    const long long uniforme = (long long)llround(pow(8.0, (double)opt.max_level));
    printf("  Refinamiento uniforme produciria: %lld cubos (reduccion: %.1f%%)\n",
           uniforme, 100.0 * (1.0 - (double)total_leaves / (double)uniforme));

    printf("\n=== Descomposicion ===\n");
    printf("  Subarboles procesados         : %d (nivel %d)\n",
           n_subtrees, split_level);
    printf("  Pasadas externas              : %d de %d\n", passes_done, max_passes);
    if (opt.uniform) {
        printf("  Balance 2:1 entre subarboles  : no aplica (modo uniforme)\n");
    } else if (converged) {
        printf("  Balance 2:1 entre subarboles  : verificado, 0 violaciones\n");
    } else {
        printf("  Balance 2:1 entre subarboles  : %llu violaciones SIN resolver\n",
               violations);
        printf("  ADVERTENCIA: sube --passes para cerrar el balance de frontera.\n");
    }
    {
        char b[32];
        printf("  Halo residente en VRAM        : %s\n",
               formatBytes(shellTableBytes(shell_prev), b, sizeof(b)));
    }

    // ---- Tabla de tiempo y espacio por nivel ----
    LevelStat rows[22];
    int n_rows = 0;
    for (int level = 1; level <= opt.max_level; level++)
        if (stats[level].n_leaves > 0 || stats[level].seconds > 0.0)
            rows[n_rows++] = stats[level];
    printLevelTable(rows, n_rows, bytes_mesh, sizeof(Cube), total_time);

    // ---- Cerrar el archivo ----
    if (vtk) {
        const long long nc = vtkStreamCount(vtk);
        if (nc > vtkMaxCubes()) {
            fprintf(stderr,
                    "\nAviso: %lld cubos superan el limite del VTK legacy "
                    "(%lld, indices de punto int32). No se escribe el archivo; "
                    "usa --no-vtk o un split-level mayor con salida por piezas.\n",
                    nc, vtkMaxCubes());
            vtkStreamDiscard(vtk);
        } else {
            char b[32];
            printf("\nEscribiendo VTK (%s)...\n",
                   formatBytes((size_t)vtkEstimatedBytes(nc), b, sizeof(b)));
            vtkStreamFinish(vtk);
        }
    }

    // ---- Limpieza ----
    shellTableFree(shell_prev);
    shellTableFree(shell_new);
    free(g_adjacent);
    CUDA_CHECK(cudaFree(d_roots));
    CUDA_CHECK(cudaFree(d_verts));
    CUDA_CHECK(cudaFree(d_tris));

    printf("\nRefinamiento completado exitosamente!\n");
    return EXIT_SUCCESS;
}
