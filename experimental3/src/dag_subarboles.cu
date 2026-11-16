#include "dag_subarboles.cuh"
#include "stats.cuh"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>

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
// Un paso uniforme: refina todas las hojas, clasifica y poda los OUTSIDE.
// Consume (libera) *d_actual y lo reemplaza por el resultado.
//
// Los picos se declaran a la contabilidad de dag.cu con el mismo criterio de
// main.cu: durante la poda coexisten la salida del refinamiento y el buffer
// de poda (peor caso); durante el refinamiento, entrada y salida.
// --------------------------------------------------------------------------
void pasoUniforme(Cube** d_actual, int* n_actual,
                  const DagClasificarFn& clasificar, size_t bytes_ajenos) {
    const int n_antes = *n_actual;
    if ((long long)n_antes * 8 > (long long)INT_MAX) {
        fprintf(stderr, "Error: un subárbol llegaría a %lld cubos (límite %d). "
                "Aumenta --split-level.\n", (long long)n_antes * 8, INT_MAX);
        exit(EXIT_FAILURE);
    }

    Cube* d_ref = nullptr;
    int n_ref = 0;
    launchRefinement(*d_actual, n_antes, &d_ref, &n_ref);
    const size_t peak_refine = ((size_t)n_antes + (size_t)n_ref) * sizeof(Cube);
    dagTrackerNota(bytes_ajenos + peak_refine);
    CUDA_CHECK(cudaFree(*d_actual));
    *d_actual = nullptr;

    clasificar(d_ref, n_ref);

    Cube* d_poda = nullptr;
    int n_poda = 0;
    launchPrune(d_ref, n_ref, &d_poda, &n_poda);
    const size_t peak_prune = 2 * (size_t)n_ref * sizeof(Cube);
    dagTrackerNota(bytes_ajenos + peak_prune);
    CUDA_CHECK(cudaFree(d_ref));

    *d_actual = d_poda;
    *n_actual = n_poda;
}

// Rehace el subárbol de la raíz *d_raiz hasta max_nivel.
void subarbolRegenerar(const Cube* d_raiz, int max_nivel,
                       const DagClasificarFn& clasificar, size_t bytes_ajenos,
                       Cube** d_hojas, int* n_hojas) {
    Cube* d_sub = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sub, sizeof(Cube)));
    CUDA_CHECK(cudaMemcpy(d_sub, d_raiz, sizeof(Cube), cudaMemcpyDeviceToDevice));

    // El nivel de la raíz se lee del propio cubo (un entero al host).
    int nivel = 0;
    CUDA_CHECK(cudaMemcpy(&nivel, reinterpret_cast<const char*>(d_raiz) +
                          offsetof(Cube, level), sizeof(int),
                          cudaMemcpyDeviceToHost));
    int n = 1;
    for (; nivel < max_nivel && n > 0; ++nivel)
        pasoUniforme(&d_sub, &n, clasificar, bytes_ajenos);
    *d_hojas = d_sub;
    *n_hojas = n;
}

void subarbolesLiberar(SubarbolesInfo* info) {
    if (info->d_raices) CUDA_CHECK(cudaFree(info->d_raices));
    info->d_raices = nullptr;
    info->n_raices = 0;
}

bool dagConstruirUniformePorSubarboles(const Cube& raiz_host,
                                       const SubarbolesConfig& cfg,
                                       Dag* dag, SubarbolesInfo* info) {
    *info = SubarbolesInfo{};
    if (cfg.split < 1 || cfg.split > cfg.max_nivel) {
        fprintf(stderr, "split-level %d fuera de rango (1..%d)\n", cfg.split,
                cfg.max_nivel);
        return false;
    }

    dagTrackerReiniciar();
    dagTrackerExterno(cfg.bytes_malla);
    gpuResetStagePeak();
    const double t0 = wallTime();

    // ---- Fase A: descenso uniforme hasta el nivel D ----
    Cube* d_cur = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cur, sizeof(Cube)));
    CUDA_CHECK(cudaMemcpy(d_cur, &raiz_host, sizeof(Cube), cudaMemcpyHostToDevice));
    int n = 1;
    cfg.clasificar(d_cur, 1);
    for (int nivel = 0; nivel < cfg.split && n > 0; ++nivel)
        pasoUniforme(&d_cur, &n, cfg.clasificar, cfg.bytes_malla);
    info->d_raices = d_cur;
    info->n_raices = n;
    const double t_a = wallTime();
    info->seg_fase_a = t_a - t0;
    if (cfg.verboso)
        printf("  Fase A: %d raíces de subárbol en el nivel %d (%.3f s)\n", n,
               cfg.split, info->seg_fase_a);

    // ---- Fase B: cada subárbol, solo, contra el DAG global ----
    DagGlobal* g = dagGlobalCrear(cfg.espejo);
    unsigned long long* d_claves = nullptr;
    uint32_t* d_slots = nullptr;
    dagReservarRefs(n, &d_claves, &d_slots);
    bool alguna_hoja = false;

    if (cfg.split == cfg.max_nivel) {
        // Las raíces ya son hojas del nivel máximo: no hay nada que refinar.
        dagEntradasDeHojas(d_cur, n, d_claves, d_slots);
        alguna_hoja = n > 0;
        info->n_subarboles = n;
        info->max_hojas_subarbol = n > 0 ? 1 : 0;
    } else {
        for (int i = 0; i < n; ++i) {
            const double t_s = wallTime();
            Cube* d_sub = nullptr;
            CUDA_CHECK(cudaMalloc(&d_sub, sizeof(Cube)));
            CUDA_CHECK(cudaMemcpy(d_sub, d_cur + i, sizeof(Cube),
                                  cudaMemcpyDeviceToDevice));
            int n_sub = 1;
            for (int nivel = cfg.split; nivel < cfg.max_nivel && n_sub > 0; ++nivel)
                pasoUniforme(&d_sub, &n_sub, cfg.clasificar, cfg.bytes_malla);

            if (n_sub > 0) {
                // Comprimir contra el DAG global; las hojas del subárbol
                // siguen vivas durante la compresión y se liberan justo
                // después.
                dagTrackerExterno(cfg.bytes_malla + (size_t)n_sub * sizeof(Cube));
                dagGlobalInsertarSubarbol(g, d_sub, n_sub, cfg.max_nivel,
                                          cfg.split, d_claves + i, d_slots + i);
                dagTrackerExterno(cfg.bytes_malla);
                alguna_hoja = true;
                ++info->n_subarboles;
                if (n_sub > info->max_hojas_subarbol) info->max_hojas_subarbol = n_sub;
            }
            CUDA_CHECK(cudaFree(d_sub));
            if (cfg.verboso)
                printf("  Fase B: subárbol %d/%d: %d hojas, comprimido y liberado "
                       "(%.3f s)\n", i + 1, n, n_sub, wallTime() - t_s);
        }
    }
    const double t_b = wallTime();
    info->seg_fase_b = t_b - t_a;

    // ---- Fase C: niveles D..1 y la raíz ----
    dagGlobalCompletar(g, cfg.split, d_claves, d_slots, n,
                       alguna_hoja ? cfg.max_nivel : -1, dag);
    dagGlobalLiberar(g);
    const double t_c = wallTime();
    info->seg_fase_c = t_c - t_b;
    info->seg_total = t_c - t0;

    dag->por_subarboles = true;
    dag->segundos = info->seg_total;
    dag->pico_calculado = dagTrackerPico();
    dag->pico_medido = gpuStagePeak();
    info->pico_calculado = dag->pico_calculado;
    info->pico_medido = dag->pico_medido;
    return true;
}

// ==========================================================================
// Hito 2b: adaptativo por subárboles con halo (mecanismo portado de
// experimental/): descenso uniforme hasta D, cada subárbol con el halo de las
// cáscaras de sus vecinos adyacentes (guardadas por paso de refinamiento y con
// la bandera split), verificación 2:1 sobre la unión de las cáscaras finales y
// pasadas hasta 0 violaciones.
//
// Diferencia con experimental/: el resultado de cada subárbol no se evacúa a
// disco; se comprime contra el DAG global de la pasada y se liberan sus hojas.
// Cada pasada construye su propio DagGlobal: si la verificación da violaciones
// se descarta y se repite; si da 0, ese DAG es el resultado.
// ==========================================================================

// Cáscaras por subárbol y por paso. La ranura max_nivel+1 guarda la final.
struct TablaCascaras {
    int    n_sub;
    int    n_slots;       // max_nivel + 2
    Cube** ptr;           // [s * n_slots + paso]
    int**  flags;
    int*   count;
};

static void tablaIniciar(TablaCascaras& t, int n_sub, int max_nivel) {
    t.n_sub = n_sub;
    t.n_slots = max_nivel + 2;
    const size_t n = (size_t)n_sub * t.n_slots;
    t.ptr   = (Cube**)calloc(n, sizeof(Cube*));
    t.flags = (int**)calloc(n, sizeof(int*));
    t.count = (int*)calloc(n, sizeof(int));
    if (!t.ptr || !t.flags || !t.count) {
        fprintf(stderr, "Error: sin memoria para la tabla de cáscaras.\n");
        exit(EXIT_FAILURE);
    }
}

static void tablaLimpiar(TablaCascaras& t) {
    const size_t n = (size_t)t.n_sub * t.n_slots;
    for (size_t i = 0; i < n; i++) {
        if (t.ptr[i])   { CUDA_CHECK(cudaFree(t.ptr[i]));   t.ptr[i] = nullptr; }
        if (t.flags[i]) { CUDA_CHECK(cudaFree(t.flags[i])); t.flags[i] = nullptr; }
        t.count[i] = 0;
    }
}

static void tablaLiberar(TablaCascaras& t) {
    tablaLimpiar(t);
    free(t.ptr); free(t.flags); free(t.count);
    t.ptr = nullptr; t.flags = nullptr; t.count = nullptr;
}

static size_t tablaBytes(const TablaCascaras& t) {
    const size_t n = (size_t)t.n_sub * t.n_slots;
    size_t total = 0;
    for (size_t i = 0; i < n; i++)
        total += (size_t)t.count[i] * (sizeof(Cube) + sizeof(int));
    return total;
}

static void tablaPonerCascara(TablaCascaras& t, int s, int slot, Cube* p,
                              int* fl, int n) {
    const size_t i = (size_t)s * t.n_slots + slot;
    if (t.ptr[i])   CUDA_CHECK(cudaFree(t.ptr[i]));
    if (t.flags[i]) CUDA_CHECK(cudaFree(t.flags[i]));
    t.ptr[i] = p;
    t.flags[i] = fl;
    t.count[i] = n;
}

static Cube* clonarCubos(const Cube* src, int n) {
    if (!src || n <= 0) return nullptr;
    Cube* dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, (size_t)n * sizeof(Cube)));
    CUDA_CHECK(cudaMemcpy(dst, src, (size_t)n * sizeof(Cube),
                          cudaMemcpyDeviceToDevice));
    return dst;
}

static int* clonarInts(const int* src, int n) {
    if (!src || n <= 0) return nullptr;
    int* dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dst, src, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    return dst;
}

// Adyacencia entre subárboles: dos cajas de nivel D se tocan solo si sus
// índices de rejilla difieren a lo más en 1 por eje. Solo las cáscaras de los
// adyacentes entran al halo.
static void construirAdyacencia(const Cube* h_raices, int n_sub,
                                std::vector<char>& ady) {
    ady.assign((size_t)n_sub * n_sub, 0);
    for (int a = 0; a < n_sub; a++) {
        for (int b = 0; b < n_sub; b++) {
            if (a == b) continue;
            const int dx = h_raices[a].grid_index.x - h_raices[b].grid_index.x;
            const int dy = h_raices[a].grid_index.y - h_raices[b].grid_index.y;
            const int dz = h_raices[a].grid_index.z - h_raices[b].grid_index.z;
            ady[(size_t)a * n_sub + b] =
                (dx >= -1 && dx <= 1) && (dy >= -1 && dy <= 1) &&
                (dz >= -1 && dz <= 1);
        }
    }
}

// Halo del subárbol `target` para el paso `slot`: las cáscaras de los
// adyacentes en ese mismo paso (lockstep). Se excluye la propia.
static void armarHalo(const TablaCascaras& t, const std::vector<char>& ady,
                      int target, int slot, Cube** d_ghost, int** d_ghost_split,
                      int* n_ghost) {
    *d_ghost = nullptr;
    *d_ghost_split = nullptr;
    *n_ghost = 0;

    long long total = 0;
    for (int s = 0; s < t.n_sub; s++) {
        if (!ady[(size_t)target * t.n_sub + s]) continue;
        total += t.count[(size_t)s * t.n_slots + slot];
    }
    if (total <= 0) return;
    if (total > INT_MAX) {
        fprintf(stderr, "Error: el halo (%lld celdas) supera el límite de int.\n", total);
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMalloc(d_ghost, (size_t)total * sizeof(Cube)));
    CUDA_CHECK(cudaMalloc(d_ghost_split, (size_t)total * sizeof(int)));
    long long offset = 0;
    for (int s = 0; s < t.n_sub; s++) {
        if (!ady[(size_t)target * t.n_sub + s]) continue;
        const size_t i = (size_t)s * t.n_slots + slot;
        const int n = t.count[i];
        if (n <= 0) continue;
        CUDA_CHECK(cudaMemcpy(*d_ghost + offset, t.ptr[i], (size_t)n * sizeof(Cube),
                              cudaMemcpyDeviceToDevice));
        if (t.flags[i])
            CUDA_CHECK(cudaMemcpy(*d_ghost_split + offset, t.flags[i],
                                  (size_t)n * sizeof(int), cudaMemcpyDeviceToDevice));
        else
            CUDA_CHECK(cudaMemset(*d_ghost_split + offset, 0, (size_t)n * sizeof(int)));
        offset += n;
    }
    *n_ghost = (int)total;
}

bool dagConstruirAdaptivePorSubarboles(const Cube& raiz_host,
                                       const SubarbolesConfig& cfg,
                                       Dag* dag, SubarbolesInfo* info) {
    *info = SubarbolesInfo{};
    if (cfg.split < 1 || cfg.split > cfg.max_nivel) {
        fprintf(stderr, "split-level %d fuera de rango (1..%d)\n", cfg.split,
                cfg.max_nivel);
        return false;
    }
    const int D = cfg.split;
    const int max_pasadas = cfg.max_pasadas > 0 ? cfg.max_pasadas : 1;

    dagTrackerReiniciar();
    dagTrackerExterno(cfg.bytes_malla);
    gpuResetStagePeak();
    const double t0 = wallTime();

    // ---- Fase A: descenso uniforme hasta el nivel D ----
    Cube* d_cur = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cur, sizeof(Cube)));
    CUDA_CHECK(cudaMemcpy(d_cur, &raiz_host, sizeof(Cube), cudaMemcpyHostToDevice));
    int n = 1;
    cfg.clasificar(d_cur, 1);
    // Limitación de la fase A: el descenso hasta D es uniforme, así que un cubo
    // INSIDE de nivel < D se subdivide aunque el monolítico lo dejaría grueso
    // (salvo que el balance 2:1 lo obligue). Se cuentan en GPU para avisar.
    long long inside_subdivididos = 0;
    for (int nivel = 0; nivel < D && n > 0; ++nivel) {
        inside_subdivididos += launchCountByState(d_cur, n, STATE_INSIDE);
        pasoUniforme(&d_cur, &n, cfg.clasificar, cfg.bytes_malla);
    }
    info->inside_subdivididos_fase_a = inside_subdivididos;
    info->d_raices = d_cur;
    info->n_raices = n;
    info->seg_fase_a = wallTime() - t0;
    if (n == 0) {
        DagGlobal* g_vacio = dagGlobalCrear(cfg.espejo);
        dagGlobalCompletar(g_vacio, D, nullptr, nullptr, 0, -1, dag);
        dagGlobalLiberar(g_vacio);
        dag->por_subarboles = true;
        info->convergio = true;
        info->seg_total = wallTime() - t0;
        return true;
    }
    if (cfg.verboso)
        printf("  Fase A: %d raíces de subárbol en el nivel %d (%.3f s)\n", n, D,
               info->seg_fase_a);

    const int n_sub = n;
    std::vector<char> ady;
    {
        // Las raíces son a lo más 8^D celdas: solo se bajan al host para
        // decidir qué subárboles se tocan.
        std::vector<Cube> h_raices(n_sub);
        CUDA_CHECK(cudaMemcpy(h_raices.data(), d_cur, (size_t)n_sub * sizeof(Cube),
                              cudaMemcpyDeviceToHost));
        construirAdyacencia(h_raices.data(), n_sub, ady);
    }

    TablaCascaras previo, nuevo;
    tablaIniciar(previo, n_sub, cfg.max_nivel);
    tablaIniciar(nuevo, n_sub, cfg.max_nivel);

    size_t max_bytes_cascaras = 0;
    auto bytes_ajenos = [&]() {
        const size_t b = tablaBytes(previo) + tablaBytes(nuevo);
        if (b > max_bytes_cascaras) max_bytes_cascaras = b;
        return cfg.bytes_malla + b;
    };

    bool convergio = false;
    unsigned long long violaciones = 0;
    int pasadas = 0;
    const double t_b0 = wallTime();

    // Perfil por fase y por pasada (--perfil). Solo en este modo se sincroniza
    // el device antes de cada lectura de reloj; en las corridas normales
    // fase() no hace nada.
    double t_marca = 0.0;
    auto fase = [&](int pasada, int f) {
        if (!cfg.perfil) return;
        CUDA_CHECK(cudaDeviceSynchronize());
        const double ahora = wallTime();
        info->perfil[pasada - 1][f] += ahora - t_marca;
        t_marca = ahora;
    };

    for (int pasada = 1; pasada <= max_pasadas; ++pasada) {
        pasadas = pasada;
        const double t_p = wallTime();
        if (cfg.perfil) { CUDA_CHECK(cudaDeviceSynchronize()); t_marca = wallTime(); }
        tablaLimpiar(nuevo);

        // Cada pasada construye su propio DAG global.
        DagGlobal* g = dagGlobalCrear(cfg.espejo);
        unsigned long long* d_claves = nullptr;
        uint32_t* d_slots = nullptr;
        dagReservarRefs(n_sub, &d_claves, &d_slots);

        std::vector<Cube*> cascara_final(n_sub, nullptr);
        std::vector<int> n_cascara_final(n_sub, 0);
        int max_hojas_sub = 0;
        int n_sub_no_vacios = 0;

        for (int s = 0; s < n_sub; ++s) {
            const double t_s = wallTime();
            Cube* d_own = clonarCubos(d_cur + s, 1);
            int n_own = 1;
            // Primera ranura de cáscara que hay que replicar al terminar: si el
            // subárbol participa en todos los pasos solo falta la final.
            int first_fill = cfg.max_nivel + 1;

            for (int paso = D + 1; paso <= cfg.max_nivel; ++paso) {
                Cube* d_ghost = nullptr;
                int* d_ghost_split = nullptr;
                int n_ghost = 0;
                armarHalo(previo, ady, s, paso, &d_ghost, &d_ghost_split, &n_ghost);
                fase(pasada, PF_HALO);

                Cube* d_ref = nullptr;
                int n_ref = 0, n_padres = 0;
                int* d_split_own = nullptr;
                const size_t ext = bytes_ajenos();
                {
                    // Pico del refinamiento: durante el ordenamiento de claves
                    // (propias + halo + split + prefijo + claves) y durante la
                    // generación de la salida.
                    const size_t n_trab = (size_t)n_own + n_ghost;
                    const size_t pico_claves = n_trab * sizeof(Cube) +
                        n_trab * (sizeof(int) + 16) + (size_t)n_own * sizeof(int);
                    dagTrackerNota(ext + pico_claves);
                }
                launchAdaptiveRefinementHalo(d_own, n_own, d_ghost, n_ghost,
                                             d_ghost_split, cfg.max_nivel,
                                             &d_ref, &n_ref, &n_padres,
                                             &d_split_own);
                {
                    const size_t n_trab = (size_t)n_own + n_ghost;
                    dagTrackerNota(ext + (n_trab + (size_t)n_ref) * sizeof(Cube) +
                                   n_trab * sizeof(int) + (size_t)n_own * sizeof(int));
                }
                fase(pasada, PF_REFINAR);

                // Cáscara de este paso: celdas de frontera con la decisión
                // que se tomó sobre ellas.
                Cube* d_shell = nullptr;
                int* d_shell_split = nullptr;
                int n_shell = 0;
                launchExtractShell(d_own, d_split_own, n_own, D, &d_shell,
                                   &d_shell_split, &n_shell);
                tablaPonerCascara(nuevo, s, paso, d_shell, d_shell_split, n_shell);

                if (d_ghost) CUDA_CHECK(cudaFree(d_ghost));
                if (d_ghost_split) CUDA_CHECK(cudaFree(d_ghost_split));
                if (d_split_own) CUDA_CHECK(cudaFree(d_split_own));
                fase(pasada, PF_CASCARAS);

                if (n_padres == 0) {
                    // Nada se divide en este paso, pero el subárbol sigue
                    // participando: en un paso posterior el halo de sus
                    // vecinos puede marcarlo (p. ej. una raíz INSIDE gruesa
                    // junto a celdas que se afinan después). Cortar aquí
                    // dejaba un punto fijo equivocado con SPLIT>1.
                    if (d_ref) CUDA_CHECK(cudaFree(d_ref));
                    fase(pasada, PF_OTRO);
                    continue;
                }
                CUDA_CHECK(cudaFree(d_own));
                fase(pasada, PF_OTRO);

                cfg.clasificar(d_ref, n_ref);
                fase(pasada, PF_CLASIFICAR);
                int n_poda = 0;
                launchPruneInPlace(d_ref, n_ref, &n_poda);
                dagTrackerNota(bytes_ajenos() + (size_t)n_ref * sizeof(Cube));
                fase(pasada, PF_PODAR);

                d_own = d_ref;
                n_own = n_poda;
                if (n_own == 0) { first_fill = paso + 1; break; }
            }

            // Cuando el subárbol deja de crecer su cáscara ya no cambia: se
            // replica en las ranuras restantes (banderas en cero) y la última
            // queda como cáscara final para la verificación.
            {
                Cube* d_tail = nullptr;
                int* d_tail_split = nullptr;
                int n_tail = 0;
                launchExtractShell(d_own, nullptr, n_own, D, &d_tail,
                                   &d_tail_split, &n_tail);
                for (int q = first_fill; q <= cfg.max_nivel + 1; ++q)
                    tablaPonerCascara(nuevo, s, q, clonarCubos(d_tail, n_tail),
                                      clonarInts(d_tail_split, n_tail), n_tail);
                if (d_tail_split) CUDA_CHECK(cudaFree(d_tail_split));
                cascara_final[s] = d_tail;
                n_cascara_final[s] = n_tail;
            }
            fase(pasada, PF_CASCARAS);

            // Comprimir el subárbol contra el DAG de esta pasada y liberar sus
            // hojas.
            if (n_own > 0) {
                dagTrackerExterno(bytes_ajenos() + (size_t)n_own * sizeof(Cube));
                dagGlobalInsertarSubarbolGeneral(g, d_own, n_own, D,
                                                 d_claves + s, d_slots + s);
                dagTrackerExterno(bytes_ajenos());
                ++n_sub_no_vacios;
                if (n_own > max_hojas_sub) max_hojas_sub = n_own;
            }
            fase(pasada, PF_INSERTAR);
            gpuSampleVram();
            if (d_own) CUDA_CHECK(cudaFree(d_own));
            if (cfg.verboso)
                printf("  Pasada %d, subárbol %2d/%d: %9d hojas | cáscara %8d | %.3f s\n",
                       pasada, s + 1, n_sub, n_own, n_cascara_final[s],
                       wallTime() - t_s);
        }

        // ---- Verificación 2:1 sobre la unión de las cáscaras finales ----
        violaciones = 0;
        long long n_union = 0;
        for (int s = 0; s < n_sub; s++) n_union += n_cascara_final[s];
        if (n_union > 0) {
            if (n_union > INT_MAX) {
                fprintf(stderr, "Error: la unión de cáscaras (%lld) supera int.\n", n_union);
                exit(EXIT_FAILURE);
            }
            dagTrackerNota(bytes_ajenos() + (size_t)n_union * (2 * sizeof(Cube) + 16));
            Cube* d_union = nullptr;
            CUDA_CHECK(cudaMalloc(&d_union, (size_t)n_union * sizeof(Cube)));
            long long off = 0;
            for (int s = 0; s < n_sub; s++) {
                if (n_cascara_final[s] <= 0) continue;
                CUDA_CHECK(cudaMemcpy(d_union + off, cascara_final[s],
                                      (size_t)n_cascara_final[s] * sizeof(Cube),
                                      cudaMemcpyDeviceToDevice));
                off += n_cascara_final[s];
            }
            violaciones = launchVerifyBalance(d_union, (int)n_union, D);
            CUDA_CHECK(cudaFree(d_union));
        }
        for (int s = 0; s < n_sub; s++)
            if (cascara_final[s]) CUDA_CHECK(cudaFree(cascara_final[s]));
        fase(pasada, PF_VERIFICAR);

        if (pasada <= 16) info->violaciones_pasada[pasada - 1] = (long long)violaciones;
        info->seg_pasada[pasada <= 16 ? pasada - 1 : 15] = wallTime() - t_p;
        info->n_subarboles = n_sub_no_vacios;
        info->max_hojas_subarbol = max_hojas_sub;
        if (cfg.verboso)
            printf("  Pasada %d: verificación 2:1 entre subárboles: %llu violaciones (%.3f s)\n\n",
                   pasada, violaciones, wallTime() - t_p);

        convergio = (violaciones == 0);
        const bool ultima = (pasada == max_pasadas);

        // Las cáscaras de esta pasada son el halo de la siguiente.
        TablaCascaras tmp = previo;
        previo = nuevo;
        nuevo = tmp;

        if (convergio || ultima) {
            // El DAG de esta pasada es el resultado (si no convergió, se
            // entrega marcado para que el llamador lo reporte).
            dagGlobalCompletar(g, D, d_claves, d_slots, n_sub, cfg.max_nivel, dag);
            dagGlobalLiberar(g);
            fase(pasada, PF_FASE_C);
            break;
        }
        // Pasada descartada: se repite con las cáscaras recién medidas.
        dagGlobalDescartar(g, d_claves, d_slots, n_sub);
        fase(pasada, PF_OTRO);
    }

    info->seg_fase_b = wallTime() - t_b0;
    info->pasadas = pasadas;
    info->convergio = convergio;
    info->violaciones_finales = (long long)violaciones;
    info->bytes_cascaras_max = max_bytes_cascaras;
    info->bytes_cascaras_final = tablaBytes(previo);
    info->seg_total = wallTime() - t0;

    dag->por_subarboles = true;
    dag->segundos = info->seg_total;
    dag->pico_calculado = dagTrackerPico();
    dag->pico_medido = gpuStagePeak();
    info->pico_calculado = dag->pico_calculado;
    info->pico_medido = dag->pico_medido;

    tablaLiberar(previo);
    tablaLiberar(nuevo);
    return true;
}

const char* nombreFasePerfil(int f) {
    static const char* nombres[PF_N] = {
        "armar halo", "refinar", "cáscaras", "clasificar", "podar",
        "insertar en DAG", "verificar cáscaras", "fase C", "otros"};
    return (f >= 0 && f < PF_N) ? nombres[f] : "?";
}
