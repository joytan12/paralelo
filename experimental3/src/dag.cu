#include "dag.cuh"
#include "stats.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/merge.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

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
// Contabilidad de VRAM de la construcción. Cada buffer propio se reserva y
// libera con estas funciones, así que el pico "calculado" es la suma máxima
// de buffers vivos (igual que stats.cu: se deriva de tamaños de buffer). No
// incluye el espacio temporal interno de thrust ni las hojas de entrada. En
// cada reserva también se muestrea el driver con gpuSampleVram().
// --------------------------------------------------------------------------
static size_t g_vivo  = 0;   // buffers propios vivos
static size_t g_pico  = 0;   // máximo de g_vivo + g_extra
static size_t g_extra = 0;   // bytes vivos ajenos (hojas, malla) declarados

// Interfaz para quien arma el DAG por partes (dag_subarboles.cu): permite
// declarar los bytes que viven fuera de este archivo (malla, hojas del
// subárbol, buffers del refinamiento) para que el pico calculado sea total.
void dagTrackerReiniciar() { g_vivo = 0; g_pico = 0; g_extra = 0; }
void dagTrackerExterno(size_t bytes) {
    g_extra = bytes;
    if (g_vivo + g_extra > g_pico) g_pico = g_vivo + g_extra;
}
void dagTrackerNota(size_t bytes_extra) {
    if (g_vivo + bytes_extra > g_pico) g_pico = g_vivo + bytes_extra;
}
size_t dagTrackerPico() { return g_pico; }
size_t dagTrackerVivo() { return g_vivo; }

template <class T>
static T* dalloc(size_t n) {
    if (n == 0) n = 1;
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    g_vivo += n * sizeof(T);
    if (g_vivo + g_extra > g_pico) g_pico = g_vivo + g_extra;
    gpuSampleVram();
    return p;
}

template <class T>
static void dfree(T* p, size_t n) {
    if (!p) return;
    if (n == 0) n = 1;
    CUDA_CHECK(cudaFree(p));
    g_vivo -= n * sizeof(T);
}

// Lee un contador entero del device (único dato que viaja al host).
template <class T>
static T leerContador(const T* d_ptr) {
    T v;
    CUDA_CHECK(cudaMemcpy(&v, d_ptr, sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

static const int THREADS = 256;
static inline int bloques(long long n) {
    return (int)((n + THREADS - 1) / THREADS);
}

// --------------------------------------------------------------------------
// Kernels de construcción.
// --------------------------------------------------------------------------

// Histograma de hojas por nivel. hist[DAG_NIVELES] cuenta niveles inválidos.
__global__ void histNivelesKernel(const Cube* cubos, int n, int* hist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int lv = cubos[i].level;
    atomicAdd(&hist[(lv >= 0 && lv <= DAG_MAX_NIVEL) ? lv : DAG_NIVELES], 1);
}

static __host__ __device__ inline uint32_t slotDeHoja(int estado) {
    return dagSlot(estado == STATE_INSIDE ? DAG_HOJA_INSIDE :
                   estado == STATE_BORDER ? DAG_HOJA_BORDER : DAG_VACIO, 0, 0);
}

struct EsNivel {
    const Cube* cubos;
    int nivel;
    __host__ __device__ bool operator()(int i) const {
        return cubos[i].level == nivel;
    }
};

// Copia clave y slot de las hojas seleccionadas a las posiciones [0, m).
__global__ void recogerHojasKernel(const Cube* cubos, const int* sel, int m,
                                   unsigned long long* claves,
                                   uint32_t* slots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const Cube& c = cubos[sel[i]];
    claves[i] = c.path_code;
    slots[i]  = slotDeHoja(c.state);
}

// Hojas contiguas por padre: marca el primer hijo de cada grupo (path>>3).
__global__ void cabezasPadreKernel(const unsigned long long* claves, int n,
                                   int* cab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cab[i] = (i == 0 || (claves[i] >> 3) != (claves[i - 1] >> 3)) ? 1 : 0;
}

// Arma los 8 slots de cada padre (hijo ausente = VACÍO = 0 por el memset).
__global__ void armarNodosKernel(const unsigned long long* claves,
                                 const uint32_t* slots, const int* pidx,
                                 const int* cab, int n, DagNode* crudo,
                                 unsigned long long* clave_padre) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = pidx[i] - 1;
    crudo[p].child[claves[i] & 7ULL] = slots[i];
    if (cab[i]) clave_padre[p] = claves[i] >> 3;
}

// Un hijo con simetría propia (estabilizador no trivial) admite varias
// máscaras equivalentes: NODO(idx, m) y NODO(idx, m ^ s) son el mismo
// subárbol si la reflexión s deja invariante al nodo idx. Para que la
// identidad de un padre dependa solo del contenido, toda máscara se
// normaliza a la menor de su clase. estab_hijos[idx] es un bitset de 8 bits:
// el bit s está activo si s deja invariante al nodo idx del nivel de los
// hijos.
static __device__ __forceinline__ uint32_t normalizarSlot(
    uint32_t s, const uint8_t* estab_hijos) {
    if (dagTipo(s) != DAG_NODO) return s;
    const uint32_t st = estab_hijos[dagIdx(s)];
    const uint32_t mk = dagMascara(s);
    uint32_t mejor = mk;
    for (uint32_t t = 1; t < 8; ++t)
        if (((st >> t) & 1u) && (mk ^ t) < mejor) mejor = mk ^ t;
    return dagSlot(DAG_NODO, mejor, dagIdx(s));
}

// Canonicalización: de las 8 reflexiones del nodo (o solo la identidad) se
// elige la lexicográficamente menor. La vista bajo m es
//     V[o] = normalizar( reflejar_m(C[o ^ m]) )
// Si la vista menor es C = V_m(O), entonces O = V_m(C) (la vista es una
// involución), por lo que la referencia a O es (idx(C), m). En empate se
// conserva la máscara menor: así el mismo contenido siempre produce la misma
// referencia, aun cuando el nodo tenga simetría propia.
//
// Además calcula el estabilizador del nodo canónico: las reflexiones s con
// V_s(C) = C. Las máscaras que empatan forman la clase m_min ^ estabilizador,
// de donde se obtiene.
__global__ void canonizarKernel(const DagNode* crudo, int n, bool espejo,
                                const uint8_t* estab_hijos,
                                DagNode* canon, uint32_t* masc,
                                uint8_t* estab_padre) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;

    const DagNode c = crudo[p];
    uint32_t mejor[8];
    uint32_t mejor_m = 0;
    uint32_t empates = 1u;                       // bit m: V_m == mejor
    for (int o = 0; o < 8; ++o) mejor[o] = c.child[o];

    if (espejo) {
        for (uint32_t m = 1; m < 8; ++m) {
            uint32_t v[8];
            int cmp = 0;                         // <0 menor, 0 igual, >0 mayor
            for (int o = 0; o < 8; ++o) {
                v[o] = normalizarSlot(dagReflejar(c.child[o ^ m], m),
                                      estab_hijos);
                if (cmp == 0 && v[o] != mejor[o]) cmp = v[o] < mejor[o] ? -1 : 1;
            }
            if (cmp < 0) {
                for (int o = 0; o < 8; ++o) mejor[o] = v[o];
                mejor_m = m;
                empates = 1u << m;
            } else if (cmp == 0) {
                empates |= 1u << m;
            }
        }
    }

    uint32_t estab = 0;
    for (uint32_t t = 0; t < 8; ++t)
        if ((empates >> t) & 1u) estab |= 1u << (t ^ mejor_m);

    for (int o = 0; o < 8; ++o) canon[p].child[o] = mejor[o];
    masc[p] = mejor_m;
    estab_padre[p] = (uint8_t)estab;
}

// Estabilizador de cada nodo único (es igual para todos los de su clase).
__global__ void estabilizadorUnicoKernel(const uint32_t* ids, const int* cab,
                                         const int* u,
                                         const uint8_t* estab_padre, int n,
                                         uint8_t* estab_unico) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !cab[i]) return;
    estab_unico[u[i] - 1] = estab_padre[ids[i]];
}

// Comparador lexicográfico exacto sobre los 8 uint32. Sin hash: la igualdad
// de identidad es la igualdad palabra a palabra.
struct NodoMenor {
    __host__ __device__ bool operator()(const DagNode& a,
                                        const DagNode& b) const {
        for (int o = 0; o < 8; ++o) {
            if (a.child[o] != b.child[o]) return a.child[o] < b.child[o];
        }
        return false;
    }
};

__global__ void cabezasUnicoKernel(const DagNode* nodos, int n, int* cab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool igual = i > 0;
    if (igual) {
        for (int o = 0; o < 8; ++o) {
            if (nodos[i].child[o] != nodos[i - 1].child[o]) { igual = false; break; }
        }
    }
    cab[i] = igual ? 0 : 1;
}

struct EsUno {
    __host__ __device__ bool operator()(int v) const { return v == 1; }
};

// Cada padre queda como NODO(idx único, máscara usada) para el nivel L-1.
__global__ void referenciasKernel(const uint32_t* ids, const int* u,
                                  const uint32_t* masc, int n,
                                  uint32_t* nuevo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t p = ids[i];
    nuevo[p] = dagSlot(DAG_NODO, masc[p], (uint32_t)(u[i] - 1));
}

__global__ void raizHojaKernel(const Cube* cubos, uint32_t* raiz) {
    raiz[0] = slotDeHoja(cubos[0].state);
}

// --------------------------------------------------------------------------
// Construcción del DAG, de abajo hacia arriba.
// --------------------------------------------------------------------------
void dagBuild(const Cube* d_cubes, int n_cubes, bool espejo, Dag* dag) {
    *dag = Dag{};
    dag->espejo = espejo;
    dag->max_nivel = -1;

    g_vivo = 0;
    g_pico = 0;
    gpuResetStagePeak();
    const double t0 = wallTime();

    dag->d_raiz = dalloc<uint32_t>(1);
    CUDA_CHECK(cudaMemset(dag->d_raiz, 0, sizeof(uint32_t)));  // VACÍO

    if (n_cubes <= 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
        dag->segundos = wallTime() - t0;
        dag->pico_calculado = g_pico;
        dag->pico_medido = gpuStagePeak();
        return;
    }

    // ---- Hojas por nivel (histograma en device, 22 enteros al host) ----
    int* d_hist = dalloc<int>(DAG_NIVELES + 1);
    CUDA_CHECK(cudaMemset(d_hist, 0, (DAG_NIVELES + 1) * sizeof(int)));
    histNivelesKernel<<<bloques(n_cubes), THREADS>>>(d_cubes, n_cubes, d_hist);
    CUDA_CHECK(cudaGetLastError());
    int hist[DAG_NIVELES + 1];
    CUDA_CHECK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));
    dfree(d_hist, DAG_NIVELES + 1);

    if (hist[DAG_NIVELES] > 0) {
        fprintf(stderr, "dagBuild: %d hojas con nivel fuera de 0..%d\n",
                hist[DAG_NIVELES], DAG_MAX_NIVEL);
        exit(EXIT_FAILURE);
    }
    int lmax = -1;
    for (int l = 0; l < DAG_NIVELES; ++l) {
        dag->n_hojas[l] = hist[l];
        if (hist[l] > 0) lmax = l;
    }
    dag->max_nivel = lmax;

    // Raíz que es una hoja: no hay nodos internos.
    if (lmax == 0) {
        if (n_cubes != 1) {
            fprintf(stderr, "dagBuild: hojas de nivel 0 repetidas (%d)\n", n_cubes);
            exit(EXIT_FAILURE);
        }
        raizHojaKernel<<<1, 1>>>(d_cubes, dag->d_raiz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        dag->segundos = wallTime() - t0;
        dag->pico_calculado = g_pico;
        dag->pico_medido = gpuStagePeak();
        return;
    }

    // Entradas que sube el nivel anterior: (clave, slot) ya ordenadas.
    unsigned long long* arrastre_claves = nullptr;
    uint32_t*           arrastre_slots  = nullptr;
    int                 n_arrastre      = 0;

    // Estabilizador de los nodos únicos del nivel L (los hijos de los nodos
    // que se construyen en la iteración L). En el nivel más profundo no hay.
    uint8_t* estab_hijos = nullptr;
    int      n_estab_hijos = 0;

    for (int L = lmax; L >= 1; --L) {
        const int n_hojas = hist[L];
        const int E = n_arrastre + n_hojas;

        // 1) Entradas del nivel L: hojas de nivel L + nodos internos de
        //    nivel L ya resueltos como referencia.
        unsigned long long* claves = dalloc<unsigned long long>(E);
        uint32_t*           slots  = dalloc<uint32_t>(E);
        if (n_arrastre > 0) {
            CUDA_CHECK(cudaMemcpy(claves, arrastre_claves,
                                  (size_t)n_arrastre * sizeof(unsigned long long),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(slots, arrastre_slots,
                                  (size_t)n_arrastre * sizeof(uint32_t),
                                  cudaMemcpyDeviceToDevice));
            dfree(arrastre_claves, n_arrastre);
            dfree(arrastre_slots, n_arrastre);
            arrastre_claves = nullptr;
            arrastre_slots = nullptr;
        }
        if (n_hojas > 0) {
            int* sel = dalloc<int>(n_hojas);
            EsNivel pred{d_cubes, L};
            thrust::copy_if(thrust::device,
                            thrust::counting_iterator<int>(0),
                            thrust::counting_iterator<int>(n_cubes),
                            sel, pred);
            recogerHojasKernel<<<bloques(n_hojas), THREADS>>>(
                d_cubes, sel, n_hojas, claves + n_arrastre, slots + n_arrastre);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            dfree(sel, n_hojas);
        }

        // 2) Agrupar por padre: ordenar por path_code deja a los hermanos
        //    contiguos (comparten path_code >> 3).
        thrust::sort_by_key(thrust::device, claves, claves + E, slots);

        int* cab  = dalloc<int>(E);
        int* pidx = dalloc<int>(E);
        cabezasPadreKernel<<<bloques(E), THREADS>>>(claves, E, cab);
        CUDA_CHECK(cudaGetLastError());
        thrust::inclusive_scan(thrust::device, cab, cab + E, pidx);
        const int n_par = leerContador(pidx + (E - 1));

        DagNode* crudo = dalloc<DagNode>(n_par);
        CUDA_CHECK(cudaMemset(crudo, 0, (size_t)n_par * sizeof(DagNode)));
        unsigned long long* clave_padre = dalloc<unsigned long long>(n_par);
        armarNodosKernel<<<bloques(E), THREADS>>>(
            claves, slots, pidx, cab, E, crudo, clave_padre);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        dfree(claves, E);
        dfree(slots, E);
        dfree(cab, E);
        dfree(pidx, E);

        // 3) Canonicalizar (menor de las 8 reflexiones, o solo identidad).
        DagNode*  canon = dalloc<DagNode>(n_par);
        uint32_t* masc  = dalloc<uint32_t>(n_par);
        uint8_t*  estab_padre = dalloc<uint8_t>(n_par);
        canonizarKernel<<<bloques(n_par), THREADS>>>(
            crudo, n_par, espejo, estab_hijos, canon, masc, estab_padre);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        dfree(crudo, n_par);
        dfree(estab_hijos, n_estab_hijos);       // ya no hace falta
        estab_hijos = nullptr;
        n_estab_hijos = 0;

        // 4) Deduplicar con sort + unique, igualdad exacta de los 8 uint32.
        uint32_t* ids = dalloc<uint32_t>(n_par);
        thrust::sequence(thrust::device, ids, ids + n_par);
        thrust::sort_by_key(thrust::device, canon, canon + n_par, ids,
                            NodoMenor());

        int* cab2 = dalloc<int>(n_par);
        int* u    = dalloc<int>(n_par);
        cabezasUnicoKernel<<<bloques(n_par), THREADS>>>(canon, n_par, cab2);
        CUDA_CHECK(cudaGetLastError());
        thrust::inclusive_scan(thrust::device, cab2, cab2 + n_par, u);
        const int n_unico = leerContador(u + (n_par - 1));
        if ((unsigned)n_unico > DAG_MASK_IDX) {
            fprintf(stderr, "dagBuild: %d nodos únicos en el nivel %d no caben "
                    "en el índice de %d bits\n", n_unico, L - 1, DAG_BITS_IDX);
            exit(EXIT_FAILURE);
        }

        dag->d_nodos[L - 1] = dalloc<DagNode>(n_unico);
        thrust::copy_if(thrust::device, canon, canon + n_par, cab2,
                        dag->d_nodos[L - 1], EsUno());

        // 5) Cada padre queda como (idx, máscara) y sube al nivel L-1.
        uint32_t* nuevo = dalloc<uint32_t>(n_par);
        referenciasKernel<<<bloques(n_par), THREADS>>>(ids, u, masc, n_par, nuevo);
        CUDA_CHECK(cudaGetLastError());
        uint8_t* estab_unico = dalloc<uint8_t>(n_unico);
        estabilizadorUnicoKernel<<<bloques(n_par), THREADS>>>(
            ids, cab2, u, estab_padre, n_par, estab_unico);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        dfree(estab_padre, n_par);
        estab_hijos = estab_unico;               // lo usa el nivel L-1
        n_estab_hijos = n_unico;
        dfree(canon, n_par);
        dfree(ids, n_par);
        dfree(cab2, n_par);
        dfree(u, n_par);
        dfree(masc, n_par);

        dag->n_logicos[L - 1] = n_par;
        dag->n_unicos[L - 1]  = n_unico;

        arrastre_claves = clave_padre;   // ya salen ordenados por clave
        arrastre_slots  = nuevo;
        n_arrastre      = n_par;
    }

    // Tras el nivel 1 solo queda el padre de nivel 0: la raíz.
    if (n_arrastre != 1) {
        fprintf(stderr, "dagBuild: se esperaba 1 raíz y hay %d (¿falta la raíz "
                "en las hojas?)\n", n_arrastre);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(dag->d_raiz, arrastre_slots, sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));
    dfree(arrastre_claves, n_arrastre);
    dfree(arrastre_slots, n_arrastre);
    dfree(estab_hijos, n_estab_hijos);

    CUDA_CHECK(cudaDeviceSynchronize());
    dag->segundos = wallTime() - t0;
    dag->pico_calculado = g_pico;
    dag->pico_medido = gpuStagePeak();
}

void dagFree(Dag* dag) {
    for (int l = 0; l < DAG_NIVELES; ++l) {
        if (dag->d_nodos[l]) CUDA_CHECK(cudaFree(dag->d_nodos[l]));
        dag->d_nodos[l] = nullptr;
    }
    if (dag->d_raiz) CUDA_CHECK(cudaFree(dag->d_raiz));
    dag->d_raiz = nullptr;
}

DagView dagView(const Dag& dag) {
    DagView v;
    v.d_raiz = dag.d_raiz;
    for (int l = 0; l < DAG_NIVELES; ++l) v.nodos[l] = dag.d_nodos[l];
    v.max_nivel = dag.max_nivel;
    return v;
}

long long dagTotalLogicos(const Dag& dag) {
    long long t = 0;
    for (int l = 0; l < DAG_NIVELES; ++l) t += dag.n_logicos[l];
    return t;
}

long long dagTotalUnicos(const Dag& dag) {
    long long t = 0;
    for (int l = 0; l < DAG_NIVELES; ++l) t += dag.n_unicos[l];
    return t;
}

// --------------------------------------------------------------------------
// Descompresión: expansión del DAG a hojas, nivel por nivel y en device.
// --------------------------------------------------------------------------
struct RegistroHoja {
    unsigned long long path;
    int nivel;
    int estado;
};

struct RegistroMenor {
    __host__ __device__ bool operator()(const RegistroHoja& a,
                                        const RegistroHoja& b) const {
        return a.path < b.path;
    }
};

static __host__ __device__ inline bool esHoja(uint32_t tipo) {
    return tipo == DAG_HOJA_INSIDE || tipo == DAG_HOJA_BORDER;
}

__global__ void iniciarFrenteKernel(const uint32_t* raiz,
                                    unsigned long long* fpath,
                                    uint32_t* fslot) {
    fpath[0] = 1ULL;
    fslot[0] = raiz[0];
}

// Por cada celda del frente: ¿es hoja? y ¿cuántos hijos no vacíos aporta?
__global__ void contarFrenteKernel(DagView v, int k, const uint32_t* fslot,
                                   int n, int* es_hoja, int* hijos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t s = fslot[i];
    const uint32_t t = dagTipo(s);
    es_hoja[i] = esHoja(t) ? 1 : 0;
    int cuenta = 0;
    if (t == DAG_NODO && k < DAG_NIVELES && v.nodos[k] != nullptr) {
        const uint32_t m = dagMascara(s);
        const DagNode& nodo = v.nodos[k][dagIdx(s)];
        for (int o = 0; o < 8; ++o)
            if (dagTipo(nodo.child[o ^ m]) != DAG_VACIO) ++cuenta;
    }
    hijos[i] = cuenta;
}

__global__ void expandirFrenteKernel(DagView v, int k,
                                     const unsigned long long* fpath,
                                     const uint32_t* fslot, int n,
                                     const int* es_hoja, const int* pos_hoja,
                                     const int* pos_hijo,
                                     RegistroHoja* hojas,
                                     unsigned long long* npath,
                                     uint32_t* nslot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t s = fslot[i];
    const uint32_t t = dagTipo(s);
    if (es_hoja[i]) {
        hojas[pos_hoja[i]] = RegistroHoja{fpath[i], k, (int)t};
        return;
    }
    if (t != DAG_NODO || k >= DAG_NIVELES || v.nodos[k] == nullptr) return;
    const uint32_t m = dagMascara(s);
    const DagNode& nodo = v.nodos[k][dagIdx(s)];
    int w = pos_hijo[i];
    for (int o = 0; o < 8; ++o) {
        // Vista: V[o] = reflejar_m(C[o ^ m])
        const uint32_t c = dagReflejar(nodo.child[o ^ m], m);
        if (dagTipo(c) == DAG_VACIO) continue;
        npath[w] = (fpath[i] << 3) | (unsigned long long)o;
        nslot[w] = c;
        ++w;
    }
}

// Expande el DAG a un arreglo de RegistroHoja en device, ordenado por path.
// El llamador libera el resultado con cudaFree.
//
// expandirDesde parte de un frente de celdas del nivel k0 (fpath, fslot, de
// las que toma posesión) y avanza hasta el nivel k_fin o hasta que no queden
// celdas. Si se piden, devuelve el frente restante al detenerse en k_fin.
static RegistroHoja* expandirDesde(const Dag& dag, int k0,
                                   unsigned long long* fpath, uint32_t* fslot,
                                   long long n_frente, int k_fin,
                                   long long* n_total,
                                   unsigned long long** frente_path = nullptr,
                                   uint32_t** frente_slot = nullptr,
                                   long long* n_frente_final = nullptr) {
    const DagView v = dagView(dag);
    std::vector<RegistroHoja*> trozos;
    std::vector<long long> n_trozos;

    int k = k0;
    for (; n_frente > 0 && k < k_fin && k <= DAG_MAX_NIVEL; ++k) {
        const int n = (int)n_frente;
        int* es_hoja  = dalloc<int>(n);
        int* hijos    = dalloc<int>(n);
        int* pos_hoja = dalloc<int>(n);
        int* pos_hijo = dalloc<int>(n);
        contarFrenteKernel<<<bloques(n), THREADS>>>(v, k, fslot, n, es_hoja, hijos);
        CUDA_CHECK(cudaGetLastError());
        thrust::exclusive_scan(thrust::device, es_hoja, es_hoja + n, pos_hoja);
        thrust::exclusive_scan(thrust::device, hijos, hijos + n, pos_hijo);
        const int n_hojas = leerContador(pos_hoja + (n - 1)) +
                            leerContador(es_hoja + (n - 1));
        const int n_sig   = leerContador(pos_hijo + (n - 1)) +
                            leerContador(hijos + (n - 1));

        RegistroHoja* hojas = n_hojas > 0 ? dalloc<RegistroHoja>(n_hojas) : nullptr;
        unsigned long long* npath = dalloc<unsigned long long>(n_sig);
        uint32_t* nslot = dalloc<uint32_t>(n_sig);
        expandirFrenteKernel<<<bloques(n), THREADS>>>(
            v, k, fpath, fslot, n, es_hoja, pos_hoja, pos_hijo,
            hojas, npath, nslot);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (n_hojas > 0) {
            trozos.push_back(hojas);
            n_trozos.push_back(n_hojas);
        }
        dfree(es_hoja, n);  dfree(hijos, n);
        dfree(pos_hoja, n); dfree(pos_hijo, n);
        dfree(fpath, n);    dfree(fslot, n);
        fpath = npath;
        fslot = nslot;
        n_frente = n_sig;
    }
    if (frente_path) {
        *frente_path = fpath;
        *frente_slot = fslot;
        *n_frente_final = n_frente;
    } else {
        dfree(fpath, n_frente > 0 ? n_frente : 1);
        dfree(fslot, n_frente > 0 ? n_frente : 1);
    }

    long long total = 0;
    for (long long c : n_trozos) total += c;
    *n_total = total;

    RegistroHoja* todas = dalloc<RegistroHoja>((size_t)total);
    long long desde = 0;
    for (size_t t = 0; t < trozos.size(); ++t) {
        CUDA_CHECK(cudaMemcpy(todas + desde, trozos[t],
                              (size_t)n_trozos[t] * sizeof(RegistroHoja),
                              cudaMemcpyDeviceToDevice));
        desde += n_trozos[t];
        dfree(trozos[t], n_trozos[t]);
    }
    if (total > 1)
        thrust::sort(thrust::device, todas, todas + total, RegistroMenor());
    return todas;
}

static RegistroHoja* expandirDag(const Dag& dag, long long* n_total) {
    unsigned long long* fpath = dalloc<unsigned long long>(1);
    uint32_t* fslot = dalloc<uint32_t>(1);
    iniciarFrenteKernel<<<1, 1>>>(dag.d_raiz, fpath, fslot);
    CUDA_CHECK(cudaGetLastError());
    return expandirDesde(dag, 0, fpath, fslot, 1, DAG_NIVELES + 1, n_total);
}

__global__ void registrosOriginalesKernel(const Cube* cubos, int n,
                                          RegistroHoja* reg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    reg[i] = RegistroHoja{cubos[i].path_code, cubos[i].level, cubos[i].state};
}

__global__ void compararRegistrosKernel(const RegistroHoja* a,
                                        const RegistroHoja* b, int n,
                                        unsigned long long* diferencias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (a[i].path != b[i].path || a[i].nivel != b[i].nivel ||
        a[i].estado != b[i].estado)
        atomicAdd(diferencias, 1ULL);
}

bool dagVerificarDescompresion(const Dag& dag, const Cube* d_cubes,
                               int n_cubes, bool verboso) {
    long long n_exp = 0;
    RegistroHoja* expandidas = expandirDag(dag, &n_exp);

    bool ok = (n_exp == (long long)n_cubes);
    unsigned long long difs = 0;
    if (ok && n_cubes > 0) {
        RegistroHoja* orig = dalloc<RegistroHoja>(n_cubes);
        registrosOriginalesKernel<<<bloques(n_cubes), THREADS>>>(
            d_cubes, n_cubes, orig);
        CUDA_CHECK(cudaGetLastError());
        if (n_cubes > 1)
            thrust::sort(thrust::device, orig, orig + n_cubes, RegistroMenor());

        unsigned long long* d_difs = dalloc<unsigned long long>(1);
        CUDA_CHECK(cudaMemset(d_difs, 0, sizeof(unsigned long long)));
        compararRegistrosKernel<<<bloques(n_cubes), THREADS>>>(
            expandidas, orig, n_cubes, d_difs);
        CUDA_CHECK(cudaGetLastError());
        difs = leerContador(d_difs);
        dfree(d_difs, 1);
        dfree(orig, n_cubes);
        ok = (difs == 0);
    }
    dfree(expandidas, n_exp > 0 ? n_exp : 1);

    if (verboso) {
        printf("  Descompresión (%s): %lld hojas expandidas vs %d originales, "
               "%llu diferencias -> %s\n",
               dag.espejo ? "con espejo" : "sin espejo", n_exp, n_cubes, difs,
               ok ? "OK" : "FALLO");
    }
    return ok;
}

// --------------------------------------------------------------------------
// Consulta directa de cada hoja original con dagLookup.
// --------------------------------------------------------------------------
__global__ void consultarHojasKernel(DagView v, const Cube* cubos, int n,
                                     unsigned long long* malas) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Cube& c = cubos[i];
    const DagHit h = dagLookup(v, c.grid_index.x, c.grid_index.y,
                               c.grid_index.z, c.level);
    const uint32_t esperado = c.state == STATE_INSIDE ? DAG_HOJA_INSIDE :
                              c.state == STATE_BORDER ? DAG_HOJA_BORDER :
                                                        DAG_VACIO;
    if (h.tipo != esperado || h.nivel != c.level) atomicAdd(malas, 1ULL);
}

long long dagVerificarConsultas(const Dag& dag, const Cube* d_cubes,
                                int n_cubes) {
    if (n_cubes <= 0) return 0;
    unsigned long long* d_malas = dalloc<unsigned long long>(1);
    CUDA_CHECK(cudaMemset(d_malas, 0, sizeof(unsigned long long)));
    consultarHojasKernel<<<bloques(n_cubes), THREADS>>>(
        dagView(dag), d_cubes, n_cubes, d_malas);
    CUDA_CHECK(cudaGetLastError());
    const long long malas = (long long)leerContador(d_malas);
    dfree(d_malas, 1);
    return malas;
}

// --------------------------------------------------------------------------
// Balance 2:1 sobre el DAG.
// --------------------------------------------------------------------------
__global__ void balanceKernel(DagView v, const RegistroHoja* hojas, int n,
                              unsigned long long* violaciones) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int nivel = hojas[i].nivel;
    if (nivel < 2) return;   // solo una hoja de nivel >= 2 puede ver una 2 más gruesa

    // Decodificar el grid desde el path_code (3 bits por nivel, raíz = 1).
    int x = 0, y = 0, z = 0;
    for (int k = 0; k < nivel; ++k) {
        const int o = (int)((hojas[i].path >> (3 * (nivel - 1 - k))) & 7ULL);
        x = (x << 1) | (o & 1);
        y = (y << 1) | ((o >> 1) & 1);
        z = (z << 1) | ((o >> 2) & 1);
    }

    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                const DagHit h = dagLookup(v, x + dx, y + dy, z + dz, nivel);
                if (esHoja(h.tipo) && h.nivel <= nivel - 2)
                    atomicAdd(violaciones, 1ULL);
            }
}

long long dagVerificarBalance(const Dag& dag, long long* n_hojas) {
    long long n = 0;
    RegistroHoja* hojas = expandirDag(dag, &n);
    if (n_hojas) *n_hojas = n;
    long long viol = 0;
    if (n > 0) {
        unsigned long long* d_viol = dalloc<unsigned long long>(1);
        CUDA_CHECK(cudaMemset(d_viol, 0, sizeof(unsigned long long)));
        balanceKernel<<<bloques(n), THREADS>>>(dagView(dag), hojas, (int)n, d_viol);
        CUDA_CHECK(cudaGetLastError());
        viol = (long long)leerContador(d_viol);
        dfree(d_viol, 1);
    }
    dfree(hojas, n > 0 ? n : 1);
    return viol;
}

// --------------------------------------------------------------------------
// Tabla por nivel y resumen de tamaños.
// --------------------------------------------------------------------------
void dagImprimirTabla(const Dag* sin_espejo, const Dag* con_espejo) {
    const Dag* ref = sin_espejo ? sin_espejo : con_espejo;
    if (!ref) return;

    printf("\n=== Compresión DAG por nivel ===\n");
    printf(" Nivel |      Hojas | Nodos lógicos | Únicos sin espejo | Únicos con espejo\n");
    printf("-------+------------+---------------+-------------------+------------------\n");

    long long t_hojas = 0, t_log = 0;
    for (int l = 0; l <= ref->max_nivel; ++l) {
        t_hojas += ref->n_hojas[l];
        t_log   += ref->n_logicos[l];
        char a[24], b[24];
        if (sin_espejo) snprintf(a, sizeof(a), "%lld", sin_espejo->n_unicos[l]);
        else            snprintf(a, sizeof(a), "-");
        if (con_espejo) snprintf(b, sizeof(b), "%lld", con_espejo->n_unicos[l]);
        else            snprintf(b, sizeof(b), "-");
        printf(" %5d | %10lld | %13lld | %17s | %17s\n",
               l, ref->n_hojas[l], ref->n_logicos[l], a, b);
    }
    printf("-------+------------+---------------+-------------------+------------------\n");
    {
        char a[24], b[24];
        if (sin_espejo) snprintf(a, sizeof(a), "%lld", dagTotalUnicos(*sin_espejo));
        else            snprintf(a, sizeof(a), "-");
        if (con_espejo) snprintf(b, sizeof(b), "%lld", dagTotalUnicos(*con_espejo));
        else            snprintf(b, sizeof(b), "-");
        printf(" Total | %10lld | %13lld | %17s | %17s\n", t_hojas, t_log, a, b);
    }

    // ---- Resumen de tamaños ----
    char s[32];
    const size_t b_base = (size_t)t_hojas * sizeof(Cube);
    const size_t b_ptr  = (size_t)t_log * sizeof(DagNode);
    printf("\n=== Tamaño del octree final ===\n");
    printf("  Base: %lld hojas x %zu B (arreglo plano)      : %s\n",
           t_hojas, sizeof(Cube), formatBytes(b_base, s, sizeof(s)));
    printf("  Árbol con punteros sin compartir (%lld nodos x %zu B) : %s\n",
           t_log, sizeof(DagNode), formatBytes(b_ptr, s, sizeof(s)));
    const Dag* dags[2] = {sin_espejo, con_espejo};
    const char* nombres[2] = {"DAG sin espejo", "DAG con espejo"};
    for (int i = 0; i < 2; ++i) {
        if (!dags[i]) continue;
        const long long u = dagTotalUnicos(*dags[i]);
        const size_t bytes = (size_t)u * sizeof(DagNode);
        printf("  %s (%lld nodos únicos x %zu B)       : %s", nombres[i], u,
               sizeof(DagNode), formatBytes(bytes, s, sizeof(s)));
        if (bytes > 0)
            printf("  (%.1fx menor que base, %.1fx menor que árbol)",
                   (double)b_base / bytes, (double)b_ptr / bytes);
        printf("\n");
    }
    printf("  (los DAG no incluyen el slot raíz de 4 B)\n");

    // ---- Construcción: tiempo y VRAM ----
    char p1[32], p2[32];
    const bool sub = (sin_espejo && sin_espejo->por_subarboles) ||
                     (con_espejo && con_espejo->por_subarboles);
    if (sub)
        printf("\n=== Construcción del DAG por subárboles (incluye el refinamiento) ===\n");
    else
        printf("\n=== Construcción del DAG (separada del refinamiento) ===\n");
    for (int i = 0; i < 2; ++i) {
        if (!dags[i]) continue;
        printf("  %-15s: tiempo %.3f s | pico calculado %s | pico medido %s\n",
               nombres[i], dags[i]->segundos,
               formatBytes(dags[i]->pico_calculado, p1, sizeof(p1)),
               formatBytes(dags[i]->pico_medido, p2, sizeof(p2)));
    }
    if (sub)
        printf("  (pico calculado = malla + buffers del refinamiento del subárbol +\n"
               "   hojas del subárbol + DAG global, sin el temporal interno de thrust;\n"
               "   pico medido = driver, de todo el proceso desde la fase A)\n");
    else
        printf("  (pico calculado = buffers propios de la construcción, sin las hojas\n"
               "   de entrada ni el temporal interno de thrust; pico medido = driver,\n"
               "   incluye hojas de entrada y los DAG ya construidos)\n");
}

// DAG global incremental (hito 2a).
#include "dag_global.inc"
