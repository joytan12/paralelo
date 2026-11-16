#include "octree.cuh"
#include "geometry.cuh"
#include "mesh.h"
#include "mesh_classify.cuh"
#include "stats.cuh"
#include "vtk_writer.h"
#include "dag.cuh"
#include "dag_subarboles.cuh"
#include "similitud.cuh"
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
    bool        no_mirror; // DAG solo con deduplicación exacta (sin espejos)
    int         split;     // nivel de descomposición (-1 = por defecto)
    bool        no_vtk;    // no exportar el VTK
    bool        no_regen;  // sin la comparación contra regeneración por bloque
    bool        comparar;  // comparar el DAG por subárboles con el monolítico
    int         passes;    // pasadas máximas de cierre del balance (adaptativo)
    bool        perfil;    // tiempo por fase y pasada (adaptativo por subárboles)
    float       esfera;    // > 0: esfera de ese radio en lugar de la malla
    bool        similitud; // hito 4: medir similitud aproximada sobre el DAG
};

// Hito 4: medición de similitud aproximada (solo con el DAG con espejo).
static void medirSimilitud(const Options& opt, const Dag* con_espejo) {
    if (!opt.similitud) return;
    if (!con_espejo) {
        printf("\nAviso: --similitud necesita el DAG con espejo; se ignora con --no-mirror.\n");
        return;
    }
    SimResultado sim;
    dagMedirSimilitud(*con_espejo, &sim);
    dagImprimirSimilitud(sim);
}

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
    printf("      --no-mirror Compresión DAG sin simetrías de espejo: solo se\n");
    printf("                  comparten subárboles idénticos. Por defecto se\n");
    printf("                  construyen ambos DAG (sin y con espejo) para\n");
    printf("                  comparar sus tamaños.\n");
    printf("      --split-level D  Construye el DAG por subárboles (uniforme o adaptativo):\n");
    printf("                  refina hasta el nivel D y luego cada subárbol por\n");
    printf("                  separado, comprimiéndolo y liberándolo antes del\n");
    printf("                  siguiente; en adaptativo con el halo de cáscaras y\n");
    printf("                  pasadas hasta 0 violaciones. 0 = camino monolítico.\n");
    printf("                  Por defecto 1. Con D>0 se construye un solo DAG (con\n");
    printf("                  espejo, o sin él con --no-mirror) y no se exporta VTK.\n");
    printf("      --passes N  Con --split-level>0 en adaptativo: pasadas máximas de cierre\n");
    printf("                  del balance entre subárboles (por defecto 8; se detiene\n");
    printf("                  en la primera con 0 violaciones).\n");
    printf("      --perfil    Con --split-level>0 en adaptativo: tiempo por fase y por\n");
    printf("                  pasada (sincroniza el device solo en este modo).\n");
    printf("      --esfera R  Reemplaza la malla por una esfera de radio R centrada en\n");
    printf("                  el origen, con raíz de half_size 1 (0 < R <= 1,7).\n");
    printf("      --similitud Mide la similitud aproximada entre nodos del DAG con\n");
    printf("                  espejo (hermanos y todo el nivel) y estima el ahorro de\n");
    printf("                  un formato delta. Solo medición; no cambia el DAG.\n");
    printf("      --no-vtk    No exporta el VTK.\n");
    printf("      --no-regen  Con --split-level>0, omite la comparación de cada\n");
    printf("                  bloque contra su regeneración desde la malla.\n");
    printf("      --comparar  Con --split-level>0, construye además el DAG\n");
    printf("                  monolítico y verifica que sea idéntico (niveles <= 9).\n");
    printf("  -h, --help      Muestra esta ayuda.\n");
}

// Devuelve false si los argumentos son inválidos.
// want_help indica que solo se pidió la ayuda.
static bool parseOptions(int argc, char* argv[], Options& opt, bool& want_help) {
    opt.max_level = 3;
    opt.mdl_path  = "cortex.mdl";
    opt.quiet     = false;
    opt.uniform   = false;
    opt.no_mirror = false;
    opt.split     = -1;
    opt.no_vtk    = false;
    opt.no_regen  = false;
    opt.comparar  = false;
    opt.passes    = 8;
    opt.perfil    = false;
    opt.esfera    = 0.0f;
    opt.similitud = false;
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
            } else if (!strcmp(arg, "--no-mirror")) {
                opt.no_mirror = true;
            } else if (!strcmp(arg, "--no-vtk")) {
                opt.no_vtk = true;
            } else if (!strcmp(arg, "--no-regen")) {
                opt.no_regen = true;
            } else if (!strcmp(arg, "--comparar")) {
                opt.comparar = true;
            } else if (!strcmp(arg, "--split-level") ||
                       !strncmp(arg, "--split-level=", 14)) {
                const char* val = nullptr;
                if (arg[13] == '=') val = arg + 14;
                else if (i + 1 < argc) val = argv[++i];
                if (!val || val[0] == '\0') {
                    fprintf(stderr, "--split-level requiere un valor\n\n");
                    return false;
                }
                opt.split = atoi(val);
                if (opt.split < 0 || opt.split > 20) {
                    fprintf(stderr, "--split-level fuera de rango: %s (válido: 0 a 20)\n\n", val);
                    return false;
                }
            } else if (!strcmp(arg, "--passes") || !strncmp(arg, "--passes=", 9)) {
                const char* val = nullptr;
                if (arg[8] == '=') val = arg + 9;
                else if (i + 1 < argc) val = argv[++i];
                if (!val || val[0] == '\0') {
                    fprintf(stderr, "--passes requiere un valor\n\n");
                    return false;
                }
                opt.passes = atoi(val);
                if (opt.passes < 1 || opt.passes > 16) {
                    fprintf(stderr, "--passes fuera de rango: %s (válido: 1 a 16)\n\n", val);
                    return false;
                }
            } else if (!strcmp(arg, "--perfil")) {
                opt.perfil = true;
            } else if (!strcmp(arg, "--similitud")) {
                opt.similitud = true;
            } else if (!strcmp(arg, "--esfera") || !strncmp(arg, "--esfera=", 9)) {
                const char* val = nullptr;
                if (arg[8] == '=') val = arg + 9;
                else if (i + 1 < argc) val = argv[++i];
                if (!val || val[0] == '\0') {
                    fprintf(stderr, "--esfera requiere un radio\n\n");
                    return false;
                }
                opt.esfera = (float)atof(val);
                if (!(opt.esfera > 0.0f && opt.esfera <= 1.7f)) {
                    fprintf(stderr, "--esfera fuera de rango: %s (válido: 0 < R <= 1,7)\n\n", val);
                    return false;
                }
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
// Clasificación: contra la malla, o contra la esfera analítica con --esfera
// (launchClassification del base, centrada en el origen).
// --------------------------------------------------------------------------
static void clasificarCubos(const Options& opt, Cube* d_cubos, int n,
                            float3* d_verts, int3* d_tris, int n_tris) {
    if (opt.esfera > 0.0f) {
        Sphere esfera;
        esfera.center = make_float3(0.f, 0.f, 0.f);
        esfera.radius = opt.esfera;
        launchClassification(d_cubos, n, esfera);
    } else {
        launchClassificationMesh(d_cubos, n, d_verts, d_tris, n_tris);
    }
}

// --------------------------------------------------------------------------
// Validaciones de un DAG contra las hojas originales, todas en GPU.
// Devuelve true si todas pasan.
// --------------------------------------------------------------------------
static bool validarDag(const Dag& dag, const Cube* d_cubes, int n_cubes,
                       bool uniform) {
    bool ok = true;

    // (a) Descompresión: hojas expandidas == hojas originales.
    if (!dagVerificarDescompresion(dag, d_cubes, n_cubes, true)) ok = false;

    // Consulta directa de cada hoja original con dagLookup.
    const long long malas = dagVerificarConsultas(dag, d_cubes, n_cubes);
    printf("  Consultas dagLookup (%s): %lld hojas incorrectas de %d -> %s\n",
           dag.espejo ? "con espejo" : "sin espejo", malas, n_cubes,
           malas == 0 ? "OK" : "FALLO");
    if (malas != 0) ok = false;

    // (b) Balance 2:1 sobre el DAG (solo tiene sentido en modo adaptativo).
    if (!uniform) {
        long long n_hojas = 0;
        const long long viol = dagVerificarBalance(dag, &n_hojas);
        printf("  Balance 2:1 en el DAG (%s): %lld violaciones en %lld hojas -> %s\n",
               dag.espejo ? "con espejo" : "sin espejo", viol, n_hojas,
               viol == 0 ? "OK" : "FALLO");
        if (viol != 0) ok = false;
    }
    return ok;
}

// --------------------------------------------------------------------------
// Referencia monolítica (solo con --comparar): rehace el octree uniforme
// completo, lo comprime con dagBuild (hito 1) y lo compara con el DAG
// construido por subárboles. Se ejecuta DESPUÉS de medir el pico del DAG por
// subárboles, para no contaminar esas mediciones.
// --------------------------------------------------------------------------
static bool compararConMonolitico(const Options& opt, const Cube& root,
                                  const DagClasificarFn& clasificar,
                                  const Dag& dag, long long n_inside,
                                  long long n_border, bool por_cobertura) {
    printf("\n=== Comparación con el DAG monolítico (referencia del hito 1) ===\n");
    Cube* d_cur = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cur, sizeof(Cube)));
    CUDA_CHECK(cudaMemcpy(d_cur, &root, sizeof(Cube), cudaMemcpyHostToDevice));
    int n = 1;
    clasificar(d_cur, 1);
    for (int nivel = 0; nivel < opt.max_level && n > 0; ++nivel) {
        if (opt.uniform) {
            pasoUniforme(&d_cur, &n, clasificar, 0);
        } else {
            // Adaptativo monolítico: el refinamiento base con balance 2:1
            // global (sin halo), que es el oráculo.
            Cube* d_ref = nullptr;
            int n_ref = 0, n_padres = 0;
            launchAdaptiveRefinement(d_cur, n, opt.max_level, &d_ref, &n_ref,
                                     &n_padres);
            if (n_padres == 0) {
                if (d_ref) CUDA_CHECK(cudaFree(d_ref));
                break;
            }
            CUDA_CHECK(cudaFree(d_cur));
            clasificar(d_ref, n_ref);
            Cube* d_poda = nullptr;
            int n_poda = 0;
            launchPrune(d_ref, n_ref, &d_poda, &n_poda);
            CUDA_CHECK(cudaFree(d_ref));
            d_cur = d_poda;
            n = n_poda;
        }
    }

    Dag ref;
    dagBuild(d_cur, n, !opt.no_mirror, &ref);
    const long long ref_in = launchCountByState(d_cur, n, STATE_INSIDE);
    const long long ref_bo = launchCountByState(d_cur, n, STATE_BORDER);
    CUDA_CHECK(cudaFree(d_cur));

    bool ok;
    bool conteos;
    if (!por_cobertura) {
        ok = dagCompararDags(dag, ref, true);
        conteos = (ref_in == n_inside && ref_bo == n_border);
        printf("  Conteos del monolítico: Dentro %lld | Borde %lld; desde el DAG por "
               "subárboles: Dentro %lld | Borde %lld -> %s\n", ref_in, ref_bo,
               n_inside, n_border, conteos ? "OK" : "FALLO");
    } else {
        // La fase A subdividió cubos INSIDE: se espera más refinamiento
        // INSIDE, no una diferencia de cobertura. Las hojas BORDER deben
        // coincidir y las INSIDE deben estar contenidas en las del monolítico.
        printf("  (la fase A subdividió cubos INSIDE: se compara por cobertura, no "
               "hoja por hoja)\n");
        ok = dagCompararCobertura(dag, ref, true, nullptr, nullptr, nullptr, nullptr);
        conteos = (ref_bo == n_border) && (n_inside >= ref_in);
        printf("  Conteos del monolítico: Dentro %lld | Borde %lld; por subárboles: "
               "Dentro %lld | Borde %lld (Borde igual y Dentro >= monolítico) -> %s\n",
               ref_in, ref_bo, n_inside, n_border, conteos ? "OK" : "FALLO");
    }
    printf("  Nodos únicos: monolítico %lld, por subárboles %lld\n",
           dagTotalUnicos(ref), dagTotalUnicos(dag));
    dagFree(&ref);
    return ok && conteos;
}

// --------------------------------------------------------------------------
// Camino por subárboles (hito 2a uniforme, hito 2b adaptativo): el arreglo completo de
// hojas no existe nunca. Ver dag_subarboles.cuh.
// --------------------------------------------------------------------------
static int ejecutarPorSubarboles(const Options& opt, const Cube& root,
                                 float3* d_verts, int3* d_tris, int n_tris,
                                 size_t bytes_mesh) {
    const DagClasificarFn clasificar = [=](Cube* d, int n) {
        clasificarCubos(opt, d, n, d_verts, d_tris, n_tris);
    };

    printf("DAG por subárboles: nivel de descomposición D=%d, %s\n", opt.split,
           opt.no_mirror ? "sin espejo (--no-mirror)" : "con espejo (X, Y, Z)");

    SubarbolesConfig cfg;
    cfg.max_nivel = opt.max_level;
    cfg.split = opt.split;
    cfg.espejo = !opt.no_mirror;
    cfg.clasificar = clasificar;
    cfg.bytes_malla = bytes_mesh;
    cfg.verboso = !opt.quiet;
    cfg.max_pasadas = opt.passes;
    cfg.perfil = opt.perfil && !opt.uniform;
    if (opt.perfil && opt.uniform)
        printf("Aviso: --perfil solo está disponible en adaptativo; se ignora.\n");

    Dag dag;
    SubarbolesInfo info;
    const bool construido = opt.uniform
        ? dagConstruirUniformePorSubarboles(root, cfg, &dag, &info)
        : dagConstruirAdaptivePorSubarboles(root, cfg, &dag, &info);
    if (!construido) return EXIT_FAILURE;

    // (b) Conteos directamente desde el DAG, por multiplicidades.
    long long n_inside = 0, n_border = 0;
    dagContarDesdeDag(&dag, &n_inside, &n_border);
    const long long n_hojas = n_inside + n_border;
    printf("\n=== Resultado final: %lld cubos sobrevivientes (contados desde el DAG) ===\n",
           n_hojas);
    printf("  Dentro: %lld | Borde: %lld\n", n_inside, n_border);
    const double uniforme = pow(8.0, opt.max_level);
    printf("  Refinamiento uniforme produciría: %.0f cubos (reducción: %.1f%%)\n",
           uniforme, 100.0 * (1.0 - (double)n_hojas / uniforme));

    char b1[32], b2[32];
    printf("\n=== Descomposición por subárboles ===\n");
    printf("  Raíces de subárbol (nivel %d): %d, no vacías: %d\n", opt.split,
           info.n_raices, info.n_subarboles);
    printf("  Subárbol más grande: %d hojas (%s de hojas en VRAM)\n",
           info.max_hojas_subarbol,
           formatBytes((size_t)info.max_hojas_subarbol * sizeof(Cube), b1, sizeof(b1)));
    if (opt.uniform) {
        printf("  Fase A %.3f s | fase B %.3f s | fase C %.3f s | total %.3f s\n",
               info.seg_fase_a, info.seg_fase_b, info.seg_fase_c, info.seg_total);
    } else {
        printf("  Pasadas externas              : %d de %d\n", info.pasadas, opt.passes);
        for (int p = 0; p < info.pasadas && p < 16; ++p)
            printf("    pasada %d: %lld violaciones entre subárboles, %.3f s\n", p + 1,
                   info.violaciones_pasada[p], info.seg_pasada[p]);
        if (info.convergio)
            printf("  Balance 2:1 entre subárboles  : verificado, 0 violaciones\n");
        else
            printf("  Balance 2:1 entre subárboles  : %lld violaciones SIN resolver "
                   "(sube --passes)\n", info.violaciones_finales);
        printf("  Cáscaras (halo) en VRAM       : máximo %s (halo previo + nuevo), "
               "residente al final %s\n",
               formatBytes(info.bytes_cascaras_max, b1, sizeof(b1)),
               formatBytes(info.bytes_cascaras_final, b2, sizeof(b2)));
        printf("  Fase A %.3f s | pasadas (fase B) %.3f s | total %.3f s\n",
               info.seg_fase_a, info.seg_fase_b, info.seg_total);
        if (info.inside_subdivididos_fase_a > 0) {
            printf("\n  AVISO (fase A): el descenso uniforme hasta el nivel %d subdividió "
                   "%lld cubos INSIDE.\n"
                   "  El resultado sigue siendo un octree válido (balance 2:1 verificado) "
                   "pero está más refinado en el interior que el monolítico, que los\n"
                   "  habría dejado gruesos; --comparar puede dar diferencias INSIDE por "
                   "ese motivo. Con --split-level 1 o 2 en cortex no ocurre.\n",
                   opt.split, info.inside_subdivididos_fase_a);
        }
        if (cfg.perfil) {
            printf("\n=== Perfil de tiempos por fase y pasada (s; el device se sincroniza "
                   "antes de cada lectura) ===\n");
            printf(" %-20s", "fase");
            for (int p = 0; p < info.pasadas && p < 16; ++p) printf(" | pasada %-2d", p + 1);
            printf(" |    total |      %%\n");
            double total_fases = 0.0;
            double por_fase[PF_N] = {};
            for (int f = 0; f < PF_N; ++f)
                for (int p = 0; p < info.pasadas && p < 16; ++p)
                    por_fase[f] += info.perfil[p][f];
            for (int f = 0; f < PF_N; ++f) total_fases += por_fase[f];
            for (int f = 0; f < PF_N; ++f) {
                printf(" %-20s", nombreFasePerfil(f));
                for (int p = 0; p < info.pasadas && p < 16; ++p)
                    printf(" | %9.3f", info.perfil[p][f]);
                printf(" | %8.3f | %5.1f\n", por_fase[f],
                       total_fases > 0 ? 100.0 * por_fase[f] / total_fases : 0.0);
            }
            printf(" %-20s", "suma de fases");
            for (int p = 0; p < info.pasadas && p < 16; ++p) {
                double s = 0.0;
                for (int f = 0; f < PF_N; ++f) s += info.perfil[p][f];
                printf(" | %9.3f", s);
            }
            printf(" | %8.3f | 100.0\n", total_fases);
        }
    }

    dagImprimirTabla(opt.no_mirror ? &dag : nullptr,
                     opt.no_mirror ? nullptr : &dag);
    medirSimilitud(opt, opt.no_mirror ? nullptr : &dag);

    printf("\n=== Validaciones del DAG (todas en GPU) ===\n");
    const double t_v = wallTime();
    long long n_bloques_hojas = 0;
    bool ok = true;
    if (opt.uniform) {
        // (c) Control por bloques: se expande el DAG subárbol por subárbol, se
        //     consulta cada hoja con dagLookup y se compara con la
        //     regeneración del subárbol desde la malla; cada bloque se libera
        //     antes del siguiente.
        DagRegenerarFn regenerar;
        if (!opt.no_regen) {
            const int max_nivel = opt.max_level;
            regenerar = [=](const Cube* d_raiz, Cube** d_h, int* n_h) {
                subarbolRegenerar(d_raiz, max_nivel, clasificar, bytes_mesh, d_h, n_h);
            };
        }
        ok = dagVerificarPorBloques(dag, opt.split, info.d_raices, info.n_raices,
                                    regenerar, true, &n_bloques_hojas);
    } else {
        // Adaptativo: balance 2:1 global sobre el DAG final, por bloques, con
        // dagLookup sobre el DAG completo (independiente del chequeo de
        // cáscaras). La regeneración por bloque no aplica: un subárbol no se
        // regenera solo sin su halo.
        long long viol = 0;
        ok = dagVerificarBloquesAdaptivo(dag, opt.split, true, &n_bloques_hojas, &viol);
        printf("  Regeneración por bloque: no aplica en adaptativo (un subárbol no se "
               "regenera solo sin halo); solo en --uniform.\n");
        if (!info.convergio) ok = false;
    }
    if (n_bloques_hojas != n_hojas) {
        printf("  Hojas recorridas por bloques: %lld, contadas por multiplicidad: "
               "%lld -> FALLO\n", n_bloques_hojas, n_hojas);
        ok = false;
    } else {
        printf("  Hojas recorridas por bloques == contadas por multiplicidad (%lld) -> OK\n",
               n_hojas);
    }
    printf("  Tiempo de las verificaciones por bloques: %.3f s\n", wallTime() - t_v);

    const bool por_cobertura = !opt.uniform && info.inside_subdivididos_fase_a > 0;
    if (opt.comparar && !compararConMonolitico(opt, root, clasificar, dag,
                                               n_inside, n_border, por_cobertura))
        ok = false;
    printf("Validación del DAG: %s\n", ok ? "CORRECTA" : "FALLÓ");

    printf("\nVTK: NO se escribe con --split-level>0 (las hojas completas nunca están "
           "en VRAM; la exportación en streaming queda para un hito posterior).\n"
           "     Usa --split-level 0 (make SPLIT=0) para el camino monolítico del "
           "hito 1, que sí exporta el VTK.\n");

    dagFree(&dag);
    subarbolesLiberar(&info);
    CUDA_CHECK(cudaFree(d_verts));
    CUDA_CHECK(cudaFree(d_tris));
    if (!ok) {
        fprintf(stderr, "\nError: la validación del DAG falló.\n");
        return EXIT_FAILURE;
    }
    printf("\n¡Refinamiento completado exitosamente!\n");
    return EXIT_SUCCESS;
}

// --------------------------------------------------------------------------
// main:
//   Flujo por etapa: refinar -> clasificar -> podar.
//   Al terminar el bucle, el octree final se comprime a un DAG con espejos
//   (dag.cu) sin tocar el refinamiento.
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

    // Nivel de descomposición: por defecto 1 (DAG por subárboles) en uniforme y
    // adaptativo; 0 es el camino monolítico del hito 1.
    if (opt.split < 0) opt.split = 1;
    if (opt.split > opt.max_level) {
        fprintf(stderr, "--split-level %d no puede superar los niveles (%d).\n",
                opt.split, opt.max_level);
        return EXIT_FAILURE;
    }

    const char* mode_name = opt.uniform ? "uniforme (todas las hojas)"
                                        : "adaptativo con balance 2:1";

    printf("=== Octree CUDA — Refinamiento con Malla + compresión DAG ===\n");
    printf("Niveles de refinamiento: %d\n", opt.max_level);
    printf("Modo de refinamiento   : %s\n", mode_name);
    if (opt.esfera > 0.0f)
        printf("Geometría: esfera de radio %.3f centrada en el origen (raíz half_size 1)\n\n",
               opt.esfera);
    else
        printf("Modelo: %s\n\n", opt.mdl_path);

    // ---- Cargar el modelo .mdl (o usar la esfera analítica con --esfera) ----
    Mesh mesh = {};
    const bool usa_esfera = opt.esfera > 0.0f;
    if (!usa_esfera) {
        if (!loadMDL(opt.mdl_path, mesh)) {
            fprintf(stderr, "Error cargando el modelo '%s'\n", opt.mdl_path);
            return EXIT_FAILURE;
        }
    } else {
        // Esfera centrada en el origen: raíz cúbica de half_size 1 (potencia
        // de 2, así los centros son diádicos y la clasificación es simétrica).
        mesh.center = make_float3(0.f, 0.f, 0.f);
        mesh.half_size = 1.0f;
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
    if (!usa_esfera) {
        CUDA_CHECK(cudaMalloc(&d_verts, mesh.n_verts * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(d_verts, mesh.vertices,
                               mesh.n_verts * sizeof(float3),
                               cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_tris, mesh.n_tris * sizeof(int3)));
        CUDA_CHECK(cudaMemcpy(d_tris, mesh.triangles,
                               mesh.n_tris * sizeof(int3),
                               cudaMemcpyHostToDevice));
    }
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
    clasificarCubos(opt, d_current, n_current, d_verts, d_tris, n_tris);
    gpuSampleVram();

    // Camino por subárboles: ejecutarPorSubarboles clasifica y refina por su
    // cuenta, a partir de la raíz en el host.
    if (opt.split > 0) {
        CUDA_CHECK(cudaFree(d_current));
        return ejecutarPorSubarboles(opt, root, d_verts, d_tris, n_tris, bytes_mesh);
    }

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
        clasificarCubos(opt, d_refined, n_refined,
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

    // ---- Compresión a DAG (en GPU, después del refinamiento) ----
    // La malla ya no hace falta: se libera para que el pico de la
    // construcción refleje solo las hojas y los buffers del DAG.
    CUDA_CHECK(cudaFree(d_verts));
    CUDA_CHECK(cudaFree(d_tris));
    d_verts = nullptr;
    d_tris  = nullptr;

    printf("\n=== Compresión del octree final a DAG (GPU) ===\n");
    printf("Modo de espejo: %s\n", opt.no_mirror ? "desactivado (--no-mirror)"
                                                 : "activado (X, Y, Z)");
    Dag dag_plano, dag_espejo;
    dagBuild(d_current, n_current, false, &dag_plano);
    if (!opt.no_mirror) dagBuild(d_current, n_current, true, &dag_espejo);
    dagImprimirTabla(&dag_plano, opt.no_mirror ? nullptr : &dag_espejo);
    medirSimilitud(opt, opt.no_mirror ? nullptr : &dag_espejo);

    printf("\n=== Validaciones del DAG (todas en GPU) ===\n");
    bool dag_ok = validarDag(dag_plano, d_current, n_current, opt.uniform);
    if (!opt.no_mirror &&
        !validarDag(dag_espejo, d_current, n_current, opt.uniform))
        dag_ok = false;
    printf("Validación del DAG: %s\n", dag_ok ? "CORRECTA" : "FALLÓ");

    // ---- Resumen global (refinar + comprimir), comparable con --split-level ----
    {
        // Pico calculado: el mayor entre el refinamiento (ya incluye la malla)
        // y la compresión (hojas finales + buffers propios del DAG).
        size_t pico_refinar = 0;
        for (int i = 0; i < n_stats; i++)
            if (stats[i].bytes_peak > pico_refinar) pico_refinar = stats[i].bytes_peak;
        const Dag& principal = opt.no_mirror ? dag_plano : dag_espejo;
        const size_t pico_dag = (size_t)n_current * sizeof(Cube) + principal.pico_calculado;
        char c1[32], c2[32];
        printf("\n=== Resumen global (monolítico, --split-level 0) ===\n");
        printf("  Tiempo total (refinar + construir DAG %s): %.3f s\n",
               opt.no_mirror ? "sin espejo" : "con espejo",
               total_time + principal.segundos);
        printf("  Pico calculado: %s | pico medido: %s\n",
               formatBytes(pico_refinar > pico_dag ? pico_refinar : pico_dag, c1, sizeof(c1)),
               formatBytes(gpuRunPeak(), c2, sizeof(c2)));
    }

    dagFree(&dag_plano);
    if (!opt.no_mirror) dagFree(&dag_espejo);

    // ---- Exportar a VTK ----
    if (!opt.no_vtk) {
        char vtk_filename[256];
        snprintf(vtk_filename, sizeof(vtk_filename),
                 "output/octree_%s_level_%d.vtk",
                 opt.uniform ? "uniform" : "pruned", opt.max_level);
        writeVTK(vtk_filename, h_cubes, n_current);
    } else {
        printf("VTK: deshabilitado (--no-vtk)\n");
    }

    // ---- Limpieza ----
    free(h_cubes);
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_verts));
    CUDA_CHECK(cudaFree(d_tris));

    if (!dag_ok) {
        fprintf(stderr, "\nError: la validación del DAG falló.\n");
        return EXIT_FAILURE;
    }
    printf("\n¡Refinamiento completado exitosamente!\n");
    return EXIT_SUCCESS;
}
