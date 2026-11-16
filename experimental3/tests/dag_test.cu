#include "dag.cuh"
#include "dag_subarboles.cuh"
#include "similitud.cuh"
#include "geometry.cuh"
#include "octree.cuh"
#include "stats.cuh"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error en %s:%d — %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err__));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// --------------------------------------------------------------------------
// Utilidades para armar octrees pequeños a mano. Solo importan level, state,
// grid_index y path_code: el DAG no usa la geometría.
// --------------------------------------------------------------------------
static unsigned long long pathDesdeGrid(int x, int y, int z, int nivel) {
    unsigned long long code = 1ULL;
    for (int bit = nivel - 1; bit >= 0; --bit) {
        const int o = ((x >> bit) & 1) | (((y >> bit) & 1) << 1) |
                      (((z >> bit) & 1) << 2);
        code = (code << 3) | (unsigned long long)o;
    }
    return code;
}

static Cube hoja(int x, int y, int z, int nivel, int estado = STATE_INSIDE) {
    Cube c{};
    c.center = make_float3(0.f, 0.f, 0.f);
    c.half_size = 1.0f / (float)(1 << nivel);
    c.level = nivel;
    c.state = estado;
    c.grid_index = make_int3(x, y, z);
    c.path_code = pathDesdeGrid(x, y, z, nivel);
    return c;
}

static Cube* subirADevice(const std::vector<Cube>& h) {
    Cube* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, (h.empty() ? 1 : h.size()) * sizeof(Cube)));
    if (!h.empty())
        CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(Cube),
                              cudaMemcpyHostToDevice));
    return d;
}

static bool g_ok = true;

static void fallo(const char* nombre, const char* fmt, long long a, long long b) {
    fprintf(stderr, "  FALLO [%s]: ", nombre);
    fprintf(stderr, fmt, a, b);
    fprintf(stderr, "\n");
    g_ok = false;
}

// Compara los nodos únicos por nivel con lo esperado. `esperado` lista los
// niveles 0..N-1 (los niveles internos); el resto debe ser 0.
static void compararUnicos(const char* nombre, const Dag& dag,
                           const std::vector<long long>& esperado) {
    for (int l = 0; l < DAG_NIVELES; ++l) {
        const long long e = l < (int)esperado.size() ? esperado[l] : 0;
        if (dag.n_unicos[l] != e) {
            char etiqueta[96];
            snprintf(etiqueta, sizeof(etiqueta), "%s %s nivel %d: únicos=%%lld esperado=%%lld",
                     nombre, dag.espejo ? "con espejo" : "sin espejo", l);
            fallo(nombre, etiqueta, dag.n_unicos[l], e);
        }
    }
}

// Construye el DAG sin y con espejo, verifica descompresión y consultas y
// compara los conteos por nivel. balance < 0 = no revisar; balance == 0 =
// debe ser 0 violaciones; balance > 0 = debe detectar al menos una.
static void correrCaso(const char* nombre, const std::vector<Cube>& h,
                       const std::vector<long long>& unicos_plano,
                       const std::vector<long long>& unicos_espejo,
                       int balance = -1) {
    Cube* d = subirADevice(h);
    const int n = (int)h.size();

    Dag plano, espejo;
    dagBuild(d, n, false, &plano);
    dagBuild(d, n, true, &espejo);

    bool ok = true;
    for (const Dag* dag : {&plano, &espejo}) {
        if (!dagVerificarDescompresion(*dag, d, n, false)) {
            fallo(nombre, "descompresión incorrecta (%lld, %lld)", dag->espejo, 0);
            ok = false;
        }
        const long long malas = dagVerificarConsultas(*dag, d, n);
        if (malas != 0) {
            fallo(nombre, "dagLookup incorrecto en %lld hojas (%lld)", malas, 0);
            ok = false;
        }
        if (balance >= 0) {
            const long long v = dagVerificarBalance(*dag, nullptr);
            if ((balance == 0 && v != 0) || (balance > 0 && v == 0)) {
                fallo(nombre, "balance: violaciones=%lld (esperado %lld)", v, balance);
                ok = false;
            }
        }
    }
    compararUnicos(nombre, plano, unicos_plano);
    compararUnicos(nombre, espejo, unicos_espejo);
    if (dagTotalUnicos(plano) < dagTotalUnicos(espejo)) {
        fallo(nombre, "espejo produjo más nodos (%lld > %lld)",
              dagTotalUnicos(espejo), dagTotalUnicos(plano));
        ok = false;
    }
    if (dagTotalLogicos(plano) != dagTotalLogicos(espejo)) {
        fallo(nombre, "nodos lógicos distintos (%lld vs %lld)",
              dagTotalLogicos(plano), dagTotalLogicos(espejo));
        ok = false;
    }

    // Falló alguna comprobación de este caso si g_ok cambió o ok es false;
    // se imprime el resumen del caso.
    printf("%s: %s | únicos sin espejo/con espejo por nivel:", nombre,
           ok && g_ok ? "OK" : "FALLO");
    for (int l = 0; l <= plano.max_nivel - 1; ++l)
        printf(" n%d=%lld/%lld", l, plano.n_unicos[l], espejo.n_unicos[l]);
    printf("\n");

    dagFree(&plano);
    dagFree(&espejo);
    CUDA_CHECK(cudaFree(d));
    if (!ok) g_ok = false;
}

// --------------------------------------------------------------------------
// Esfera centrada en el origen: root de half_size = 1 (potencia de 2), de
// modo que todos los centros son diádicos y la clasificación es simétrica.
// Reproduce el bucle refinar -> clasificar -> podar del programa base.
// --------------------------------------------------------------------------
static void correrEsfera(const char* nombre, bool uniforme, int max_nivel,
                         float radio) {
    Cube raiz{};
    raiz.center = make_float3(0.f, 0.f, 0.f);
    raiz.half_size = 1.0f;
    raiz.level = 0;
    raiz.state = STATE_UNCLASSIFIED;
    raiz.grid_index = make_int3(0, 0, 0);
    raiz.path_code = 1ULL;

    Sphere esfera;
    esfera.center = make_float3(0.f, 0.f, 0.f);
    esfera.radius = radio;

    Cube* d_actual = subirADevice({raiz});
    int n_actual = 1;
    launchClassification(d_actual, n_actual, esfera);

    for (int pasada = 1; pasada <= max_nivel && n_actual > 0; ++pasada) {
        Cube* d_ref = nullptr;
        int n_ref = 0, n_padres = 0;
        if (uniforme) {
            launchRefinement(d_actual, n_actual, &d_ref, &n_ref);
            n_padres = n_actual;
        } else {
            launchAdaptiveRefinement(d_actual, n_actual, max_nivel, &d_ref,
                                     &n_ref, &n_padres);
        }
        if (n_padres == 0) {
            if (d_ref) CUDA_CHECK(cudaFree(d_ref));
            break;
        }
        CUDA_CHECK(cudaFree(d_actual));
        launchClassification(d_ref, n_ref, esfera);
        Cube* d_poda = nullptr;
        int n_poda = 0;
        launchPrune(d_ref, n_ref, &d_poda, &n_poda);
        CUDA_CHECK(cudaFree(d_ref));
        d_actual = d_poda;
        n_actual = n_poda;
    }

    Dag plano, espejo;
    dagBuild(d_actual, n_actual, false, &plano);
    dagBuild(d_actual, n_actual, true, &espejo);

    bool ok = true;
    for (const Dag* dag : {&plano, &espejo}) {
        if (!dagVerificarDescompresion(*dag, d_actual, n_actual, true)) ok = false;
        const long long malas = dagVerificarConsultas(*dag, d_actual, n_actual);
        if (malas != 0) { fallo(nombre, "dagLookup incorrecto en %lld hojas (%lld)", malas, 0); ok = false; }
        if (!uniforme) {
            long long nh = 0;
            const long long v = dagVerificarBalance(*dag, &nh);
            printf("  Balance 2:1 (%s): %lld violaciones en %lld hojas\n",
                   dag->espejo ? "con espejo" : "sin espejo", v, nh);
            if (v != 0) { fallo(nombre, "balance: %lld violaciones (%lld)", v, 0); ok = false; }
        }
    }

    // Con espejo todos los octantes de la raíz son reflejos de uno solo.
    if (espejo.n_unicos[1] != 1) {
        fallo(nombre, "con espejo se esperaba 1 nodo único en el nivel 1 y hay %lld (%lld)",
              espejo.n_unicos[1], 0);
        ok = false;
    }
    if (plano.n_unicos[1] != 8) {
        fallo(nombre, "sin espejo se esperaban 8 nodos únicos en el nivel 1 y hay %lld (%lld)",
              plano.n_unicos[1], 0);
        ok = false;
    }
    for (int l = 0; l < DAG_NIVELES; ++l) {
        if (espejo.n_unicos[l] > plano.n_unicos[l]) {
            fallo(nombre, "espejo produjo más nodos únicos (%lld > %lld)",
                  espejo.n_unicos[l], plano.n_unicos[l]);
            ok = false;
        }
    }

    printf("%s: %s (hojas=%d, nivel máximo=%d)\n", nombre, ok ? "OK" : "FALLO",
           n_actual, plano.max_nivel);
    dagImprimirTabla(&plano, &espejo);
    printf("\n");

    dagFree(&plano);
    dagFree(&espejo);
    CUDA_CHECK(cudaFree(d_actual));
    if (!ok) g_ok = false;
}

// --------------------------------------------------------------------------
// Hito 2a: fusión de subárboles en el DAG global.
//
// Las hojas se parten por su ancestro de nivel `split`, cada bloque se
// comprime contra el DAG global (capacidad inicial mínima para forzar el
// crecimiento) y se completan los niveles superiores. El resultado debe ser
// idéntico en contenido al DAG monolítico del hito 1 (dagBuild).
// --------------------------------------------------------------------------
static void construirIncremental(const std::vector<Cube>& todas, int nivel_hojas,
                                 int split, bool espejo, Dag* dag) {
    std::map<unsigned long long, std::vector<Cube>> bloques;
    const int desplazamiento = 3 * (nivel_hojas - split);
    for (const Cube& c : todas) bloques[c.path_code >> desplazamiento].push_back(c);

    DagGlobal* g = dagGlobalCrear(espejo, 2);
    unsigned long long* d_claves = nullptr;
    uint32_t* d_slots = nullptr;
    dagReservarRefs((int)bloques.size(), &d_claves, &d_slots);
    int i = 0;
    for (auto& kv : bloques) {
        Cube* d = subirADevice(kv.second);
        dagGlobalInsertarSubarbol(g, d, (int)kv.second.size(), nivel_hojas, split,
                                  d_claves + i, d_slots + i);
        CUDA_CHECK(cudaFree(d));
        ++i;
    }
    dagGlobalCompletar(g, split, d_claves, d_slots, (int)bloques.size(),
                       todas.empty() ? -1 : nivel_hojas, dag);
    dagGlobalLiberar(g);
}

// Compara, para cada split y con/sin espejo, el DAG incremental con el
// monolítico, la descompresión contra las hojas, las consultas y los conteos
// por multiplicidad. unicos_espejo / unicos_plano (si no están vacíos) fijan
// los nodos únicos esperados por nivel interno.
static void correrFusion(const char* nombre, const std::vector<Cube>& todas,
                         int nivel_hojas, int split_max,
                         const std::vector<long long>& unicos_plano = {},
                         const std::vector<long long>& unicos_espejo = {}) {
    Cube* d = subirADevice(todas);
    const int n = (int)todas.size();
    long long esp_in = 0, esp_bo = 0;
    for (const Cube& c : todas) (c.state == STATE_INSIDE ? esp_in : esp_bo)++;

    bool ok = true;
    for (int split = 1; split <= split_max; ++split) {
        for (int espejo = 0; espejo < 2; ++espejo) {
            Dag inc, mono;
            construirIncremental(todas, nivel_hojas, split, espejo != 0, &inc);
            dagBuild(d, n, espejo != 0, &mono);

            if (!dagCompararDags(inc, mono, false)) {
                fallo(nombre, "split %lld: DAG incremental distinto del monolítico (espejo=%lld)",
                      split, espejo);
                ok = false;
            }
            if (!dagVerificarDescompresion(inc, d, n, false)) {
                fallo(nombre, "split %lld: descompresión incorrecta (espejo=%lld)", split, espejo);
                ok = false;
            }
            const long long malas = dagVerificarConsultas(inc, d, n);
            if (malas != 0) {
                fallo(nombre, "dagLookup incorrecto en %lld hojas (split %lld)", malas, split);
                ok = false;
            }
            long long c_in = 0, c_bo = 0;
            dagContarDesdeDag(&inc, &c_in, &c_bo);
            if (c_in != esp_in || c_bo != esp_bo) {
                fallo(nombre, "conteo por multiplicidad: dentro=%lld esperado=%lld", c_in, esp_in);
                ok = false;
            }
            const std::vector<long long>& esperado = espejo ? unicos_espejo : unicos_plano;
            if (!esperado.empty()) {
                for (int l = 0; l < DAG_NIVELES; ++l) {
                    const long long e = l < (int)esperado.size() ? esperado[l] : 0;
                    if (inc.n_unicos[l] != e) {
                        fallo(nombre, "nivel %lld: nodos únicos = %lld", l, inc.n_unicos[l]);
                        fprintf(stderr, "     (espejo=%d split=%d esperado=%lld)\n", espejo, split, e);
                        ok = false;
                    }
                }
            }
            dagFree(&inc);
            dagFree(&mono);
        }
    }
    printf("%s: %s (%d hojas, split 1..%d, con y sin espejo)\n", nombre,
           ok ? "OK" : "FALLO", n, split_max);
    CUDA_CHECK(cudaFree(d));
    if (!ok) g_ok = false;
}

// Subárbol de un octante de nivel 1: celdas de nivel 3 (4x4x4 locales) con un
// patrón pseudoaleatorio fijado por `semilla`, reflejado por `mascara`.
static unsigned hashPatron(unsigned semilla, unsigned c) {
    unsigned h = semilla * 2654435761u ^ (c + 1u) * 40503u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}

static void agregarPatron(std::vector<Cube>& out, int ox, int oy, int oz,
                          unsigned semilla, int mascara) {
    for (unsigned c = 0; c < 64; ++c) {
        const unsigned h = hashPatron(semilla, c);
        if (h % 3 == 0) continue;
        const int estado = (h % 7 == 0) ? STATE_BORDER : STATE_INSIDE;
        int lx = c & 3, ly = (c >> 2) & 3, lz = (c >> 4) & 3;
        if (mascara & 1) lx = 3 - lx;
        if (mascara & 2) ly = 3 - ly;
        if (mascara & 4) lz = 3 - lz;
        out.push_back(hoja(ox * 4 + lx, oy * 4 + ly, oz * 4 + lz, 3, estado));
    }
}

static void agregarOctante(std::vector<Cube>& out, int oct, unsigned semilla,
                           int mascara) {
    agregarPatron(out, oct & 1, (oct >> 1) & 1, (oct >> 2) & 1, semilla, mascara);
}

// Esfera centrada con el flujo por subárboles (dag_subarboles.cu) y con el
// monolítico; deben dar el mismo DAG y 1 nodo único en el nivel 1 con espejo.
static void correrEsferaSubarboles(const char* nombre, int max_nivel, int split,
                                   float radio) {
    Sphere esfera;
    esfera.center = make_float3(0.f, 0.f, 0.f);
    esfera.radius = radio;
    const DagClasificarFn clasificar = [esfera](Cube* d, int n) {
        launchClassification(d, n, esfera);
    };

    Cube raiz{};
    raiz.center = make_float3(0.f, 0.f, 0.f);
    raiz.half_size = 1.0f;
    raiz.level = 0;
    raiz.state = STATE_UNCLASSIFIED;
    raiz.grid_index = make_int3(0, 0, 0);
    raiz.path_code = 1ULL;

    // Referencia monolítica uniforme.
    Cube* d_ref = subirADevice({raiz});
    int n_ref = 1;
    clasificar(d_ref, 1);
    for (int i = 0; i < max_nivel && n_ref > 0; ++i)
        pasoUniforme(&d_ref, &n_ref, clasificar, 0);

    bool ok = true;
    Dag inc_plano, inc_espejo, mono_plano, mono_espejo;
    SubarbolesInfo info_p, info_e;
    for (int espejo = 0; espejo < 2; ++espejo) {
        SubarbolesConfig cfg;
        cfg.max_nivel = max_nivel;
        cfg.split = split;
        cfg.espejo = espejo != 0;
        cfg.clasificar = clasificar;
        Dag& inc = espejo ? inc_espejo : inc_plano;
        SubarbolesInfo& info = espejo ? info_e : info_p;
        Dag& mono = espejo ? mono_espejo : mono_plano;

        if (!dagConstruirUniformePorSubarboles(raiz, cfg, &inc, &info)) ok = false;
        dagBuild(d_ref, n_ref, espejo != 0, &mono);

        if (!dagCompararDags(inc, mono, false)) {
            fallo(nombre, "DAG por subárboles distinto del monolítico (espejo=%lld, split=%lld)", espejo, split);
            ok = false;
        }
        long long c_in = 0, c_bo = 0;
        dagContarDesdeDag(&inc, &c_in, &c_bo);
        const long long r_in = launchCountByState(d_ref, n_ref, STATE_INSIDE);
        const long long r_bo = launchCountByState(d_ref, n_ref, STATE_BORDER);
        if (c_in != r_in || c_bo != r_bo) {
            fallo(nombre, "conteos por multiplicidad (%lld) distintos de los de las hojas (%lld)", c_in + c_bo, r_in + r_bo);
            ok = false;
        }
        DagRegenerarFn regen = [=](const Cube* d_raiz, Cube** dh, int* nh) {
            subarbolRegenerar(d_raiz, max_nivel, clasificar, 0, dh, nh);
        };
        long long n_blk = 0;
        if (!dagVerificarPorBloques(inc, split, info.d_raices, info.n_raices, regen,
                                    false, &n_blk) || n_blk != c_in + c_bo) {
            fallo(nombre, "verificación por bloques falló (%lld hojas, esperado %lld)", n_blk, c_in + c_bo);
            ok = false;
        }
        const long long esperado_n1 = espejo ? 1 : 8;
        if (inc.n_unicos[1] != esperado_n1) {
            fallo(nombre, "nivel 1: %lld nodos únicos, esperado %lld", inc.n_unicos[1], esperado_n1);
            ok = false;
        }
    }
    printf("%s: %s (nivel %d, split %d; nivel 1: %lld sin espejo / %lld con espejo; "
           "únicos totales %lld / %lld)\n", nombre, ok ? "OK" : "FALLO", max_nivel,
           split, inc_plano.n_unicos[1], inc_espejo.n_unicos[1],
           dagTotalUnicos(inc_plano), dagTotalUnicos(inc_espejo));
    subarbolesLiberar(&info_p);
    subarbolesLiberar(&info_e);
    dagFree(&inc_plano); dagFree(&inc_espejo);
    dagFree(&mono_plano); dagFree(&mono_espejo);
    CUDA_CHECK(cudaFree(d_ref));
    if (!ok) g_ok = false;
}

// --------------------------------------------------------------------------
// Hito 2b: adaptativo por subárboles con halo. El oráculo es el monolítico
// (launchAdaptiveRefinement sin halo + dagBuild). Se exige: mismas hojas y
// mismos nodos únicos por nivel, conteos por multiplicidad, balance 2:1 global
// por bloques igual a 0 y, si se pide, que haya hecho falta más de una pasada.
// --------------------------------------------------------------------------
static void correrEsferaAdaptiva(const char* nombre, int max_nivel, int split,
                                 float radio, float3 centro_esfera,
                                 bool centrada, bool exigir_segunda_pasada) {
    Sphere esfera;
    esfera.center = centro_esfera;
    esfera.radius = radio;
    const DagClasificarFn clasificar = [esfera](Cube* d, int n) {
        launchClassification(d, n, esfera);
    };

    Cube raiz{};
    raiz.center = make_float3(0.f, 0.f, 0.f);
    raiz.half_size = 1.0f;
    raiz.level = 0;
    raiz.state = STATE_UNCLASSIFIED;
    raiz.grid_index = make_int3(0, 0, 0);
    raiz.path_code = 1ULL;

    // Oráculo: adaptativo monolítico.
    Cube* d_ref = subirADevice({raiz});
    int n_ref = 1;
    clasificar(d_ref, 1);
    for (int i = 0; i < max_nivel && n_ref > 0; ++i) {
        Cube* d_new = nullptr;
        int n_new = 0, n_padres = 0;
        launchAdaptiveRefinement(d_ref, n_ref, max_nivel, &d_new, &n_new, &n_padres);
        if (n_padres == 0) {
            if (d_new) CUDA_CHECK(cudaFree(d_new));
            break;
        }
        CUDA_CHECK(cudaFree(d_ref));
        clasificar(d_new, n_new);
        Cube* d_poda = nullptr;
        int n_poda = 0;
        launchPrune(d_new, n_new, &d_poda, &n_poda);
        CUDA_CHECK(cudaFree(d_new));
        d_ref = d_poda;
        n_ref = n_poda;
    }

    bool ok = true;
    Dag inc[2] = {}, mono[2] = {};
    SubarbolesInfo info[2];
    for (int espejo = 0; espejo < 2; ++espejo) {
        SubarbolesConfig cfg;
        cfg.max_nivel = max_nivel;
        cfg.split = split;
        cfg.espejo = espejo != 0;
        cfg.clasificar = clasificar;
        cfg.max_pasadas = 8;

        if (!dagConstruirAdaptivePorSubarboles(raiz, cfg, &inc[espejo], &info[espejo])) {
            fallo(nombre, "no se pudo construir el DAG (espejo=%lld, split=%lld)", espejo, split);
            ok = false;
            continue;
        }
        dagBuild(d_ref, n_ref, espejo != 0, &mono[espejo]);

        if (!info[espejo].convergio) {
            fallo(nombre, "no convergió: %lld violaciones tras %lld pasadas",
                  info[espejo].violaciones_finales, info[espejo].pasadas);
            ok = false;
        }
        if (!dagCompararDags(inc[espejo], mono[espejo], false)) {
            fallo(nombre, "DAG por subárboles distinto del monolítico (espejo=%lld, split=%lld)", espejo, split);
            ok = false;
        }
        long long c_in = 0, c_bo = 0;
        dagContarDesdeDag(&inc[espejo], &c_in, &c_bo);
        const long long r_in = launchCountByState(d_ref, n_ref, STATE_INSIDE);
        const long long r_bo = launchCountByState(d_ref, n_ref, STATE_BORDER);
        if (c_in != r_in || c_bo != r_bo) {
            fallo(nombre, "conteos por multiplicidad (%lld) distintos de los de las hojas (%lld)", c_in + c_bo, r_in + r_bo);
            ok = false;
        }
        long long n_blk = 0, viol = 0;
        if (!dagVerificarBloquesAdaptivo(inc[espejo], split, false, &n_blk, &viol) ||
            n_blk != c_in + c_bo) {
            fallo(nombre, "balance 2:1 global por bloques: %lld violaciones (esperado 0, hojas %lld)", viol, n_blk);
            ok = false;
        }
        if (centrada) {
            const long long esperado_n1 = espejo ? 1 : 8;
            if (inc[espejo].n_unicos[1] != esperado_n1) {
                fallo(nombre, "nivel 1: %lld nodos únicos, esperado %lld", inc[espejo].n_unicos[1], esperado_n1);
                ok = false;
            }
        }
        if (exigir_segunda_pasada && info[espejo].pasadas < 2) {
            fallo(nombre, "se esperaba una segunda pasada y solo hubo %lld (espejo=%lld)", info[espejo].pasadas, espejo);
            ok = false;
        }
    }
    printf("%s: %s (nivel %d, split %d; pasadas %d, violaciones por pasada:", nombre,
           ok ? "OK" : "FALLO", max_nivel, split, info[1].pasadas);
    for (int p = 0; p < info[1].pasadas && p < 16; ++p)
        printf(" %lld", info[1].violaciones_pasada[p]);
    printf("; únicos %lld / %lld)\n", dagTotalUnicos(inc[0]), dagTotalUnicos(inc[1]));
    for (int e = 0; e < 2; ++e) {
        subarbolesLiberar(&info[e]);
        dagFree(&inc[e]);
        dagFree(&mono[e]);
    }
    CUDA_CHECK(cudaFree(d_ref));
    if (!ok) g_ok = false;
}

// --------------------------------------------------------------------------
// Hito 3: limitación de la fase A. Cuando el descenso uniforme hasta D
// subdivide cubos INSIDE que el monolítico deja gruesos, el DAG por subárboles
// es un octree válido pero más refinado en el interior. Se exige: aviso con
// conteo > 0 (o == 0 si no debe haberlo), balance global 0, hojas BORDER
// idénticas al monolítico y cada hoja INSIDE contenida en una INSIDE del
// monolítico con el mismo volumen total.
// --------------------------------------------------------------------------
static void correrFaseAInside(const char* nombre, int max_nivel, int split,
                              float radio, bool esperar_aviso,
                              bool esperar_mas_inside = false) {
    Sphere esfera;
    esfera.center = make_float3(0.f, 0.f, 0.f);
    esfera.radius = radio;
    const DagClasificarFn clasificar = [esfera](Cube* d, int n) {
        launchClassification(d, n, esfera);
    };

    Cube raiz{};
    raiz.center = make_float3(0.f, 0.f, 0.f);
    raiz.half_size = 1.0f;
    raiz.level = 0;
    raiz.state = STATE_UNCLASSIFIED;
    raiz.grid_index = make_int3(0, 0, 0);
    raiz.path_code = 1ULL;

    // Oráculo: adaptativo monolítico.
    Cube* d_ref = subirADevice({raiz});
    int n_ref = 1;
    clasificar(d_ref, 1);
    for (int i = 0; i < max_nivel && n_ref > 0; ++i) {
        Cube* d_new = nullptr;
        int n_new = 0, n_padres = 0;
        launchAdaptiveRefinement(d_ref, n_ref, max_nivel, &d_new, &n_new, &n_padres);
        if (n_padres == 0) {
            if (d_new) CUDA_CHECK(cudaFree(d_new));
            break;
        }
        CUDA_CHECK(cudaFree(d_ref));
        clasificar(d_new, n_new);
        Cube* d_poda = nullptr;
        int n_poda = 0;
        launchPrune(d_new, n_new, &d_poda, &n_poda);
        CUDA_CHECK(cudaFree(d_new));
        d_ref = d_poda;
        n_ref = n_poda;
    }
    Dag mono = {};
    dagBuild(d_ref, n_ref, true, &mono);

    bool ok = true;
    Dag inc = {};
    SubarbolesInfo info;
    SubarbolesConfig cfg;
    cfg.max_nivel = max_nivel;
    cfg.split = split;
    cfg.espejo = true;
    cfg.clasificar = clasificar;
    cfg.max_pasadas = 8;
    if (!dagConstruirAdaptivePorSubarboles(raiz, cfg, &inc, &info)) {
        fallo(nombre, "no se pudo construir el DAG (split=%lld, nivel=%lld)", split, max_nivel);
        ok = false;
    } else {
        const long long avisos = info.inside_subdivididos_fase_a;
        if (esperar_aviso && avisos <= 0) {
            fallo(nombre, "se esperaba aviso de la fase A (conteo %lld > 0, split=%lld)", avisos, split);
            ok = false;
        }
        if (!esperar_aviso && avisos != 0) {
            fallo(nombre, "no se esperaba aviso de la fase A y hay %lld cubos INSIDE (split=%lld)", avisos, split);
            ok = false;
        }
        if (!info.convergio) {
            fallo(nombre, "no convergió: %lld violaciones tras %lld pasadas",
                  info.violaciones_finales, info.pasadas);
            ok = false;
        }
        long long c_in = 0, c_bo = 0;
        dagContarDesdeDag(&inc, &c_in, &c_bo);
        long long n_blk = 0, viol = 0;
        if (!dagVerificarBloquesAdaptivo(inc, split, false, &n_blk, &viol) ||
            n_blk != c_in + c_bo) {
            fallo(nombre, "balance 2:1 global por bloques: %lld violaciones (esperado 0, hojas %lld)", viol, n_blk);
            ok = false;
        }
        long long difs_borde = 0, no_cont = 0, vol_s = 0, vol_r = 0;
        if (!dagCompararCobertura(inc, mono, false, &difs_borde, &no_cont, &vol_s, &vol_r)) {
            fallo(nombre, "cobertura distinta del monolítico: %lld diferencias BORDER, %lld INSIDE no contenidas",
                  difs_borde, no_cont);
            ok = false;
        }
        const long long r_in = launchCountByState(d_ref, n_ref, STATE_INSIDE);
        const long long r_bo = launchCountByState(d_ref, n_ref, STATE_BORDER);
        if (esperar_mas_inside && c_in <= r_in) {
            fallo(nombre, "se esperaban más hojas INSIDE que el monolítico (%lld vs %lld)", c_in, r_in);
            ok = false;
        }
        if (c_bo != r_bo || c_in < r_in) {
            fallo(nombre, "conteos: Borde %lld (monolítico %lld)", c_bo, r_bo);
            ok = false;
        }
        printf("%s: %s (nivel %d, split %d; cubos INSIDE subdivididos en la fase A: %lld; "
               "hojas INSIDE %lld vs %lld del monolítico, BORDER %lld vs %lld; pasadas %d)\n",
               nombre, ok ? "OK" : "FALLO", max_nivel, split, avisos, c_in, r_in, c_bo,
               r_bo, info.pasadas);
    }
    subarbolesLiberar(&info);
    dagFree(&inc);
    dagFree(&mono);
    CUDA_CHECK(cudaFree(d_ref));
    if (!ok) g_ok = false;
}

// --------------------------------------------------------------------------
// Hito 4: medición de similitud. Devuelve, para los niveles 1 y 2:
//   [0..3]  nivel 1: M1 d=1, M2 d1 canónico, deltas conservadores, M4 tramo <=5 %
//   [4..7]  nivel 2: M1 d=1, M2 d1 canónico, deltas conservadores, M4 tramo <=25 %
//   [8..10] slots NODO de los niveles 0, 1 y 2
//   [11]    nivel 1: M1 sin hermano comparable     [12] nivel 1: únicos
// --------------------------------------------------------------------------
static std::vector<long long> medirSim(const std::vector<Cube>& h) {
    Cube* d = subirADevice(h);
    Dag dag;
    dagBuild(d, (int)h.size(), true, &dag);
    SimResultado r;
    dagMedirSimilitud(dag, &r);
    std::vector<long long> v = {
        r.niv[1].m1[1], r.niv[1].d1_canon, r.niv[1].deltas_cons, r.niv[1].m4[3],
        r.niv[2].m1[1], r.niv[2].d1_canon, r.niv[2].deltas_cons, r.niv[2].m4[5],
        r.niv[0].slots_nodo, r.niv[1].slots_nodo, r.niv[2].slots_nodo,
        r.niv[1].m1[0], r.niv[1].n};
    dagFree(&dag);
    CUDA_CHECK(cudaFree(d));
    return v;
}

static void esperarSim(const char* nombre, const std::vector<long long>& v,
                       const std::vector<long long>& esperado) {
    bool ok = true;
    for (size_t i = 0; i < esperado.size(); ++i)
        if (v[i] != esperado[i]) {
            char etiqueta[96];
            snprintf(etiqueta, sizeof(etiqueta), "medida %zu: %%lld (esperado %%lld)", i);
            fallo(nombre, etiqueta, v[i], esperado[i]);
            ok = false;
        }
    printf("%s: %s\n", nombre, ok ? "OK" : "FALLO");
}

int main() {
    CUDA_CHECK(cudaFree(0));
    gpuSetBaseline();

    // --- Caso 1: 8 hojas iguales (hijos de la raíz). ---
    {
        std::vector<Cube> h;
        for (int o = 0; o < 8; ++o) h.push_back(hoja(o & 1, (o >> 1) & 1, (o >> 2) & 1, 1));
        // Un solo nodo (la raíz), sin compartir nada más.
        correrCaso("8 hojas iguales", h, {1}, {1});
    }

    // --- Caso 2: nodos que solo difieren por reflexión en X, Y, Z y XYZ. ---
    // Cada hijo de la raíz tiene una sola hoja de nivel 2 en la posición
    // reflejada. Sin espejo son 5 nodos distintos; con espejo, 1.
    {
        std::vector<Cube> h = {
            hoja(0, 0, 0, 2), hoja(3, 0, 0, 2), hoja(0, 3, 0, 2),
            hoja(0, 0, 3, 2), hoja(3, 3, 3, 2)};
        correrCaso("simetría de espejo", h, {1, 5}, {1, 1});
    }

    // --- Caso 2b: espejo con un hijo que tiene simetría propia. ---
    // P (octante 0 de la raíz) tiene un nodo de 8 hojas, simétrico en X/Y/Z,
    // en su octante 0 y una hoja de nivel 2 en el octante 1. Q (octante 1) es
    // el espejo en X de P. Sin normalizar la máscara del hijo simétrico
    // (NODO(lleno, 0) frente a NODO(lleno, 1)) P y Q no se reconocerían.
    {
        std::vector<Cube> h;
        for (int o = 0; o < 8; ++o) {
            h.push_back(hoja(o & 1, (o >> 1) & 1, (o >> 2) & 1, 3));        // P: lleno en celda (0,0,0)@2
            h.push_back(hoja(6 + (o & 1), (o >> 1) & 1, (o >> 2) & 1, 3));  // Q: lleno en celda (3,0,0)@2
        }
        h.push_back(hoja(1, 0, 0, 2));   // P: hoja en el octante local 1
        h.push_back(hoja(2, 0, 0, 2));   // Q: hoja en el octante local 0
        // Nivel 2: el nodo lleno es uno solo; nivel 1: P y Q se unen con espejo.
        correrCaso("espejo con hijo simétrico", h, {1, 2, 1}, {1, 1, 1});
    }

    // --- Caso 3: dos hermanos que difieren en una sola rama. ---
    // A y B (octantes 0 y 1 de la raíz) tienen 8 nodos de nivel 2 con 8
    // hojas INSIDE cada uno; en B una hoja del nodo local 5 es BORDER.
    {
        std::vector<Cube> h;
        for (int ab = 0; ab < 2; ++ab) {
            for (int o2 = 0; o2 < 8; ++o2) {
                for (int o3 = 0; o3 < 8; ++o3) {
                    const int x = (ab << 2) | ((o2 & 1) << 1) | (o3 & 1);
                    const int y = (((o2 >> 1) & 1) << 1) | ((o3 >> 1) & 1);
                    const int z = (((o2 >> 2) & 1) << 1) | ((o3 >> 2) & 1);
                    const int estado = (ab == 1 && o2 == 5 && o3 == 3)
                                           ? STATE_BORDER : STATE_INSIDE;
                    h.push_back(hoja(x, y, z, 3, estado));
                }
            }
        }
        // Nivel 2: el nodo todo-INSIDE y el que tiene un BORDER; nivel 1: A y B.
        correrCaso("hermanos con una rama distinta", h, {1, 2, 2}, {1, 2, 2});
    }

    // --- Caso 4: balance 2:1 — detección de violaciones. ---
    {
        // Hoja de nivel 1 junto a hojas de nivel 2: diferencia de 1, balanceado.
        std::vector<Cube> bien = {hoja(0, 0, 0, 1), hoja(2, 0, 0, 2)};
        correrCaso("balance correcto", bien, {1, 1}, {1, 1}, 0);

        // Hoja de nivel 1 junto a una de nivel 3: diferencia de 2, violación.
        std::vector<Cube> mal = {hoja(0, 0, 0, 1), hoja(4, 0, 0, 3)};
        correrCaso("balance violado (control negativo)", mal, {1, 1, 1}, {1, 1, 1}, 1);
    }

    // --- Caso 5: raíz hoja y árbol vacío. ---
    {
        std::vector<Cube> raiz = {hoja(0, 0, 0, 0)};
        correrCaso("raíz hoja", raiz, {}, {});
        std::vector<Cube> vacio;
        correrCaso("árbol vacío", vacio, {}, {});
    }

    // --- Caso 6: esfera centrada, uniforme y adaptativa. ---
    correrEsfera("esfera uniforme (nivel 4)", true, 4, 0.7f);
    correrEsfera("esfera adaptativa (nivel 5)", false, 5, 0.7f);

    // --- Caso 7 (hito 2a): fusión de subárboles en el DAG global. ---
    {
        // Dos subárboles idénticos (octantes 0 y 1, mismo patrón): un solo
        // nodo único en el nivel 1 con y sin espejo.
        std::vector<Cube> h;
        agregarOctante(h, 0, 11, 0);
        agregarOctante(h, 1, 11, 0);
        correrFusion("fusión: subárboles idénticos", h, 3, 2);
        // (esperados fijos para split=1; el barrido split<=2 compara con el
        //  monolítico en todos los casos)

        // Subárboles espejados en X: dos nodos sin espejo, uno con espejo.
        std::vector<Cube> e;
        agregarOctante(e, 0, 11, 0);
        agregarOctante(e, 1, 11, 1);
        correrFusion("fusión: subárboles espejados", e, 3, 2);

        // Uno que aporta nodos nuevos a un nivel ya poblado: A, B, A, C (C
        // comparte casi todo con A) en octantes 0..3.
        std::vector<Cube> n;
        agregarOctante(n, 0, 11, 0);
        agregarOctante(n, 1, 23, 0);
        agregarOctante(n, 2, 11, 0);
        agregarOctante(n, 3, 37, 0);
        agregarOctante(n, 4, 23, 5);
        correrFusion("fusión: nodos nuevos en nivel poblado", n, 3, 2);
    }

    // Esperados concretos del caso de subárboles idénticos/espejados con
    // split=1 (nodos únicos del nivel 1).
    {
        std::vector<Cube> h;
        agregarOctante(h, 0, 11, 0);
        agregarOctante(h, 1, 11, 0);
        correrFusion("fusión: idénticos (únicos fijos)", h, 3, 1,
                     {1, 1, 8}, {1, 1, 8});
    }

    // --- Caso 8: aleatorio. Octantes con patrones pseudoaleatorios, algunos
    //     reflejados y algunos vacíos, split 1..3, con y sin espejo. ---
    {
        unsigned estado_rng = 12345u;
        auto sig = [&estado_rng]() {
            estado_rng = estado_rng * 1664525u + 1013904223u;
            return estado_rng >> 8;
        };
        for (int ensayo = 0; ensayo < 12; ++ensayo) {
            std::vector<Cube> h;
            for (int oct = 0; oct < 8; ++oct) {
                if (sig() % 8 == 0) continue;           // octante vacío
                agregarOctante(h, oct, 1 + sig() % 3, (int)(sig() % 8));
            }
            char nombre[64];
            snprintf(nombre, sizeof(nombre), "fusión aleatoria %d", ensayo);
            correrFusion(nombre, h, 3, 3);
        }
    }

    // --- Caso 9: esfera centrada por subárboles, split 1 y 2. ---
    correrEsferaSubarboles("esfera por subárboles (nivel 4)", 4, 1, 0.7f);
    correrEsferaSubarboles("esfera por subárboles (nivel 4)", 4, 2, 0.7f);
    correrEsferaSubarboles("esfera por subárboles (nivel 5)", 5, 2, 0.7f);

    // --- Caso 10 (hito 2b): adaptativo por subárboles con halo. ---
    {
        const float3 origen = make_float3(0.f, 0.f, 0.f);
        // Esfera centrada: nivel 1 = 1 nodo único con espejo; idéntico al
        // monolítico; balance global 0.
        correrEsferaAdaptiva("esfera adaptativa por subárboles (nivel 5)", 5, 1,
                             0.7f, origen, true, false);
        correrEsferaAdaptiva("esfera adaptativa por subárboles (nivel 5)", 5, 2,
                             0.7f, origen, true, false);
        correrEsferaAdaptiva("esfera adaptativa por subárboles (nivel 6)", 6, 2,
                             0.7f, origen, true, false);
        // Esfera grande: hay raíces INSIDE gruesas en el nivel de descomposición
        // junto a celdas que se afinan después; el halo de los vecinos debe
        // poder marcarlas aunque no se dividan en su primer paso (regresión del
        // corte anticipado del bucle de pasos).
        correrEsferaAdaptiva("esfera grande con raíces INSIDE (nivel 6)", 6, 2,
                             0.95f, origen, true, false);
        correrEsferaAdaptiva("esfera grande con raíces INSIDE (nivel 7)", 7, 2,
                             0.95f, make_float3(0.1f, -0.05f, 0.07f), false, false);
        // Esfera descentrada: la frontera entre subárboles propaga marcas y
        // hace falta una segunda pasada.
        correrEsferaAdaptiva("esfera descentrada, segunda pasada (nivel 6)", 6, 1,
                             0.55f, make_float3(0.13f, 0.06f, -0.09f), false, true);
        correrEsferaAdaptiva("esfera descentrada, segunda pasada (nivel 6)", 6, 2,
                             0.55f, make_float3(0.13f, 0.06f, -0.09f), false, true);
    }

    // --- Caso 11 (hito 3): limitación de la fase A (INSIDE sobre D). ---
    {
        // Con D=3 el descenso uniforme subdivide los 8 cubos INSIDE del nivel 2:
        // debe avisar, mantener balance global 0 y cubrir lo mismo.
        correrFaseAInside("fase A con INSIDE subdivididos (D=3, nivel 6)", 6, 3, 0.95f, true);
        correrFaseAInside("fase A con INSIDE subdivididos (D=3, nivel 7)", 7, 3, 0.95f, true);
        // Con D=2 no se subdivide ningún INSIDE: sin aviso.
        correrFaseAInside("fase A sin INSIDE subdivididos (D=2, nivel 6)", 6, 2, 0.95f, false);
        // Con D=4 hay INSIDE del nivel 3 que el monolítico deja gruesos: aquí
        // el DAG por subárboles tiene estrictamente más hojas INSIDE y la
        // comparación hoja por hoja daría diferencias; la cobertura sigue
        // siendo idéntica.
        correrFaseAInside("fase A con más INSIDE que el monolítico (D=4, nivel 7)", 7, 4, 0.95f, true, true);
    }

    // --- Caso 12 (hito 4): medición de similitud con respuesta conocida. ---
    {
        // Mismo árbol del caso 3: A y B (hijos de la raíz) tienen 8 nodos de
        // nivel 2; en B uno de ellos (G) tiene una hoja BORDER, el resto es el
        // nodo todo-INSIDE (F). A y B difieren en 1 slot; F y G también.
        std::vector<Cube> h;
        for (int ab = 0; ab < 2; ++ab)
            for (int o2 = 0; o2 < 8; ++o2)
                for (int o3 = 0; o3 < 8; ++o3) {
                    const int x = (ab << 2) | ((o2 & 1) << 1) | (o3 & 1);
                    const int y = (((o2 >> 1) & 1) << 1) | ((o3 >> 1) & 1);
                    const int z = (((o2 >> 2) & 1) << 1) | ((o3 >> 2) & 1);
                    const int estado = (ab == 1 && o2 == 5 && o3 == 3)
                                           ? STATE_BORDER : STATE_INSIDE;
                    h.push_back(hoja(x, y, z, 3, estado));
                }
        std::vector<long long> v = medirSim(h);
        // Por nivel 1 y 2: M1 d=1, M2 d1 canónico, deltas conservadores,
        // y el tramo de M4 (nivel 1: 8 de 512 celdas -> <=5 %; nivel 2: 64 de
        // 512 -> <=25 %). Slots NODO: nivel 0 = 2, nivel 1 = 16, nivel 2 = 0.
        esperarSim("similitud: hermanos a distancia 1", v,
                   {2, 2, 1, 2, 2, 2, 1, 2, 2, 16, 0});

        // Hermanos espejados (caso 2): ya comparten nodo, no hay pares.
        std::vector<Cube> e = {
            hoja(0, 0, 0, 2), hoja(3, 0, 0, 2), hoja(0, 3, 0, 2),
            hoja(0, 0, 3, 2), hoja(3, 3, 3, 2)};
        std::vector<long long> w = medirSim(e);
        // Nivel 1: 1 nodo único, sin hermano comparable y sin vecino.
        if (w[0] != 0 || w[1] != 0 || w[11] != 1 || w[12] != 1) {
            fallo("similitud: hermanos espejados", "M1 d1=%lld, sin hermano=%lld",
                  w[0], w[11]);
        } else {
            printf("similitud: hermanos espejados: OK\n");
        }
    }

    if (!g_ok) {
        fprintf(stderr, "\nHubo pruebas del DAG que fallaron.\n");
        return EXIT_FAILURE;
    }
    printf("Todas las pruebas del DAG pasaron.\n");
    return EXIT_SUCCESS;
}
