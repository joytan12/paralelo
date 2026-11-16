#include "similitud.cuh"

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define SIM_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error en %s:%d — %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err__));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

static const int SIM_BLOQUE     = 256;
static const unsigned SIM_SIN   = 0xFFFFFFFFu;   // "sin hermano comparable"
static const int SIM_PALABRAS   = 32;            // 512 celdas x 2 bits / 32

static int simBloques(long long n) {
    return (int)((n + SIM_BLOQUE - 1) / SIM_BLOQUE);
}

// --------------------------------------------------------------------------
// Normalización de un slot: la máscara de un NODO se lleva a la menor de su
// clase m ^ Stab(hijo), para que dos referencias equivalentes sean iguales
// bit a bit. stab puede ser nullptr si el nivel de los hijos no tiene nodos.
// --------------------------------------------------------------------------
__device__ __forceinline__ uint32_t simNorm(uint32_t s, const uint8_t* stab) {
    if (dagTipo(s) != DAG_NODO || stab == nullptr) return s;
    const uint32_t st = stab[dagIdx(s)];
    const uint32_t m = dagMascara(s);
    uint32_t mejor = m;
    for (uint32_t t = 1; t < 8; ++t)
        if ((st >> t) & 1u) {
            const uint32_t c = m ^ t;
            if (c < mejor) mejor = c;
        }
    return dagSlot(DAG_NODO, mejor, dagIdx(s));
}

// Slot o de la vista del nodo c bajo la reflexión m, normalizado.
__device__ __forceinline__ uint32_t simVista(const DagNode& c, int o, uint32_t m,
                                             const uint8_t* stab) {
    return simNorm(dagReflejar(c.child[o ^ m], m), stab);
}

// Estabilizador de cada nodo: bitset de las reflexiones que lo dejan igual.
__global__ void simStabKernel(const DagNode* nodos, int n,
                              const uint8_t* stab_hijos, uint8_t* stab) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const DagNode c = nodos[i];
    uint32_t base[8];
    for (int o = 0; o < 8; ++o) base[o] = simNorm(c.child[o], stab_hijos);
    uint32_t st = 1u;
    for (uint32_t m = 1; m < 8; ++m) {
        bool igual = true;
        for (int o = 0; o < 8 && igual; ++o)
            igual = simVista(c, o, m, stab_hijos) == base[o];
        if (igual) st |= 1u << m;
    }
    stab[i] = (uint8_t)st;
}

// Distancia en slots entre dos nodos, mínimo sobre las 8 reflexiones de b.
__device__ __forceinline__ unsigned simDistancia(const DagNode& a, const DagNode& b,
                                                 const uint8_t* stab) {
    uint32_t na[8];
    for (int o = 0; o < 8; ++o) na[o] = simNorm(a.child[o], stab);
    unsigned mejor = 9;
    for (uint32_t m = 0; m < 8; ++m) {
        unsigned d = 0;
        for (int o = 0; o < 8; ++o) d += simVista(b, o, m, stab) != na[o];
        if (d < mejor) mejor = d;
    }
    return mejor;
}

// M1: por cada padre, pares de hijos NODO con índice distinto.
__global__ void simM1Kernel(const DagNode* padres, int n_padres,
                            const DagNode* hijos, const uint8_t* stab_nietos,
                            unsigned* mejor) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_padres) return;
    const DagNode P = padres[p];
    for (int a = 0; a < 8; ++a) {
        if (dagTipo(P.child[a]) != DAG_NODO) continue;
        const uint32_t ia = dagIdx(P.child[a]);
        for (int b = a + 1; b < 8; ++b) {
            if (dagTipo(P.child[b]) != DAG_NODO) continue;
            const uint32_t ib = dagIdx(P.child[b]);
            if (ia == ib) continue;                 // ya compartidos
            const unsigned d = simDistancia(hijos[ia], hijos[ib], stab_nietos);
            atomicMin(&mejor[ia], d);
            atomicMin(&mejor[ib], d);
        }
    }
}

// M4: rejilla de 8x8x8 celdas (3 niveles por debajo del nodo), 2 bits por
// celda con el tipo de lo que cubre la celda (VACÍO, INSIDE, BORDER o NODO si
// sigue subdividido más abajo). Un hilo por (nodo, palabra de 16 celdas).
__global__ void simRejillaKernel(const DagNode* n0, long long n,
                                 const DagNode* n1, const DagNode* n2,
                                 uint32_t* rejilla) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * SIM_PALABRAS) return;
    const long long nodo = t / SIM_PALABRAS;
    const int w = (int)(t % SIM_PALABRAS);
    uint32_t palabra = 0;
    for (int j = 0; j < 16; ++j) {
        const int c = w * 16 + j;
        const int cx = c & 7, cy = (c >> 3) & 7, cz = (c >> 6) & 7;
        int o = ((cx >> 2) & 1) | (((cy >> 2) & 1) << 1) | (((cz >> 2) & 1) << 2);
        uint32_t s = n0[nodo].child[o];
        if (dagTipo(s) == DAG_NODO && n1 != nullptr) {
            uint32_t acum = dagMascara(s);
            o = ((cx >> 1) & 1) | (((cy >> 1) & 1) << 1) | (((cz >> 1) & 1) << 2);
            s = n1[dagIdx(s)].child[o ^ acum];
            if (dagTipo(s) == DAG_NODO && n2 != nullptr) {
                acum ^= dagMascara(s);
                o = (cx & 1) | ((cy & 1) << 1) | ((cz & 1) << 2);
                s = n2[dagIdx(s)].child[o ^ acum];
            }
        }
        palabra |= dagTipo(s) << (2 * j);
    }
    rejilla[t] = palabra;
}

__device__ __forceinline__ uint32_t simCelda(const uint32_t* g, int c) {
    return (g[c >> 4] >> ((c & 15) * 2)) & 3u;
}

// M4: celdas distintas entre las rejillas de dos hermanos, mínimo sobre las 8
// reflexiones (reflejar un nodo invierte la coordenada de cada eje marcado).
__global__ void simM4Kernel(const DagNode* padres, int n_padres,
                            const uint32_t* rejilla, unsigned* mejor) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_padres) return;
    const DagNode P = padres[p];
    for (int a = 0; a < 8; ++a) {
        if (dagTipo(P.child[a]) != DAG_NODO) continue;
        const uint32_t ia = dagIdx(P.child[a]);
        const uint32_t* ga = rejilla + (size_t)ia * SIM_PALABRAS;
        for (int b = a + 1; b < 8; ++b) {
            if (dagTipo(P.child[b]) != DAG_NODO) continue;
            const uint32_t ib = dagIdx(P.child[b]);
            if (ia == ib) continue;
            const uint32_t* gb = rejilla + (size_t)ib * SIM_PALABRAS;
            unsigned menor = 513;
            for (int m = 0; m < 8; ++m) {
                const int inv = ((m & 1) ? 7 : 0) | ((m & 2) ? 56 : 0) |
                                ((m & 4) ? 448 : 0);
                unsigned dif = 0;
                for (int c = 0; c < 512; ++c)
                    dif += simCelda(ga, c) != simCelda(gb, c ^ inv);
                if (dif < menor) menor = dif;
            }
            atomicMin(&mejor[ia], menor);
            atomicMin(&mejor[ib], menor);
        }
    }
}

// --------------------------------------------------------------------------
// M2: registros (8 slots normalizados + id del nodo) para ordenar.
// --------------------------------------------------------------------------
struct SimRec {
    uint32_t k[8];
    uint32_t id;
};

struct SimRecMenor {
    __host__ __device__ bool operator()(const SimRec& a, const SimRec& b) const {
        for (int o = 0; o < 8; ++o) {
            if (a.k[o] < b.k[o]) return true;
            if (a.k[o] > b.k[o]) return false;
        }
        return a.id < b.id;
    }
};

__device__ __forceinline__ bool simMismaClave(const SimRec& a, const SimRec& b) {
    for (int o = 0; o < 8; ++o)
        if (a.k[o] != b.k[o]) return false;
    return true;
}

// Un hilo por (nodo, vista): vista v del nodo, normalizada.
__global__ void simVistasKernel(const DagNode* nodos, long long n, int n_vistas,
                                const uint8_t* stab_hijos, SimRec* out) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * n_vistas) return;
    const long long i = t / n_vistas;
    const uint32_t v = (uint32_t)(t % n_vistas);
    const DagNode c = nodos[i];
    SimRec r;
    for (int o = 0; o < 8; ++o) r.k[o] = simVista(c, o, v, stab_hijos);
    r.id = (uint32_t)i;
    out[t] = r;
}

// Copia los registros anulando una o dos posiciones de slot (p2 < 0 = una).
__global__ void simEnmascararKernel(const SimRec* in, long long n, int p1, int p2,
                                    SimRec* out) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    SimRec r = in[t];
    r.k[p1] = 0;
    if (p2 >= 0) r.k[p2] = 0;
    out[t] = r;
}

// Con los registros ordenados por (clave, id): un nodo tiene un vecino si en
// su grupo de clave hay otro id. Basta mirar al anterior y al siguiente.
// cabezas (opcional) marca el primer registro de cada grupo.
__global__ void simMarcarKernel(const SimRec* r, long long n, uint8_t* marca,
                                int* cabezas) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const bool igual_prev = t > 0 && simMismaClave(r[t], r[t - 1]);
    const bool igual_sig = t + 1 < n && simMismaClave(r[t], r[t + 1]);
    if ((igual_prev && r[t - 1].id != r[t].id) ||
        (igual_sig && r[t + 1].id != r[t].id))
        marca[r[t].id] = 1;
    if (cabezas) cabezas[t] = igual_prev ? 0 : 1;
}

struct SimContarNodo {
    __host__ __device__ long long operator()(const DagNode& c) const {
        long long s = 0;
        for (int o = 0; o < 8; ++o) s += dagTipo(c.child[o]) == DAG_NODO;
        return s;
    }
};

struct SimEnRango {
    unsigned lo, hi;   // lo <= v <= hi
    __host__ __device__ bool operator()(unsigned v) const { return v >= lo && v <= hi; }
};

struct SimSoloD2 {
    __host__ __device__ long long operator()(const thrust::tuple<uint8_t, uint8_t>& t) const {
        return (thrust::get<1>(t) && !thrust::get<0>(t)) ? 1 : 0;
    }
};

// --------------------------------------------------------------------------
// Seguimiento simple del pico de VRAM de la medición.
// --------------------------------------------------------------------------
static size_t g_sim_libre0 = 0;
static size_t g_sim_pico = 0;

static void simMuestra() {
    size_t libre = 0, total = 0;
    SIM_CHECK(cudaMemGetInfo(&libre, &total));
    if (g_sim_libre0 > libre && g_sim_libre0 - libre > g_sim_pico)
        g_sim_pico = g_sim_libre0 - libre;
}

static long long simContar(const unsigned* d, long long n, unsigned lo, unsigned hi) {
    thrust::device_ptr<const unsigned> b(d);
    SimEnRango pred{lo, hi};
    return (long long)thrust::count_if(thrust::device, b, b + n, pred);
}

// M2 de un nivel con n_vistas = 1 (formas canónicas) u 8 (todas las vistas).
// Devuelve nodos con vecino a distancia 1 y a distancia 2 (sin vecino a 1).
// Con n_vistas = 1 calcula además los deltas de la cota conservadora.
static void simM2(const DagNode* d_nodos, long long n, const uint8_t* stab_hijos,
                  int n_vistas, long long* d1, long long* d2, long long* deltas_cons) {
    *d1 = *d2 = 0;
    if (deltas_cons) *deltas_cons = 0;
    if (n < 2) return;

    const long long nr = n * n_vistas;
    SimRec *d_base = nullptr, *d_trab = nullptr;
    uint8_t *d_m1 = nullptr, *d_m2 = nullptr;
    int* d_cab = nullptr;
    SIM_CHECK(cudaMalloc(&d_base, nr * sizeof(SimRec)));
    SIM_CHECK(cudaMalloc(&d_trab, nr * sizeof(SimRec)));
    SIM_CHECK(cudaMalloc(&d_m1, n));
    SIM_CHECK(cudaMalloc(&d_m2, n));
    SIM_CHECK(cudaMemset(d_m1, 0, n));
    SIM_CHECK(cudaMemset(d_m2, 0, n));
    if (deltas_cons) SIM_CHECK(cudaMalloc(&d_cab, nr * sizeof(int)));

    simVistasKernel<<<simBloques(nr), SIM_BLOQUE>>>(d_nodos, n, n_vistas,
                                                    stab_hijos, d_base);
    SIM_CHECK(cudaGetLastError());
    simMuestra();

    thrust::device_ptr<SimRec> tb(d_trab);
    long long mejor_deltas = 0;

    // Distancia 1: se anula una posición.
    for (int p = 0; p < 8; ++p) {
        simEnmascararKernel<<<simBloques(nr), SIM_BLOQUE>>>(d_base, nr, p, -1, d_trab);
        SIM_CHECK(cudaGetLastError());
        thrust::sort(thrust::device, tb, tb + nr, SimRecMenor());
        simMarcarKernel<<<simBloques(nr), SIM_BLOQUE>>>(d_trab, nr, d_m1, d_cab);
        SIM_CHECK(cudaGetLastError());
        if (deltas_cons) {
            thrust::device_ptr<int> c(d_cab);
            const long long grupos =
                thrust::reduce(thrust::device, c, c + nr, (long long)0);
            if (n - grupos > mejor_deltas) mejor_deltas = n - grupos;
        }
    }
    simMuestra();

    // Distancia 2: se anulan dos posiciones.
    for (int p = 0; p < 8; ++p) {
        for (int q = p + 1; q < 8; ++q) {
            simEnmascararKernel<<<simBloques(nr), SIM_BLOQUE>>>(d_base, nr, p, q, d_trab);
            SIM_CHECK(cudaGetLastError());
            thrust::sort(thrust::device, tb, tb + nr, SimRecMenor());
            simMarcarKernel<<<simBloques(nr), SIM_BLOQUE>>>(d_trab, nr, d_m2, nullptr);
            SIM_CHECK(cudaGetLastError());
        }
    }
    SIM_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<uint8_t> a(d_m1), b(d_m2);
    *d1 = thrust::reduce(thrust::device, a, a + n, (long long)0);
    *d2 = thrust::transform_reduce(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(a, b)),
        thrust::make_zip_iterator(thrust::make_tuple(a + n, b + n)),
        SimSoloD2(), (long long)0, thrust::plus<long long>());
    if (deltas_cons) *deltas_cons = mejor_deltas;

    SIM_CHECK(cudaFree(d_base));
    SIM_CHECK(cudaFree(d_trab));
    SIM_CHECK(cudaFree(d_m1));
    SIM_CHECK(cudaFree(d_m2));
    if (d_cab) SIM_CHECK(cudaFree(d_cab));
}

void dagMedirSimilitud(const Dag& dag, SimResultado* r) {
    memset(r, 0, sizeof(*r));
    r->max_nivel = dag.max_nivel;
    const auto t0 = std::chrono::steady_clock::now();

    size_t total = 0;
    SIM_CHECK(cudaMemGetInfo(&g_sim_libre0, &total));
    g_sim_pico = 0;

    // Niveles con nodos internos: 0 .. max_nivel-1.
    const int n_niv = dag.max_nivel > 0 ? dag.max_nivel : 0;
    auto cuenta = [&](int k) -> long long {
        return (k >= 0 && k < n_niv) ? dag.n_unicos[k] : 0;
    };
    auto nodos = [&](int k) -> const DagNode* {
        return cuenta(k) > 0 ? dag.d_nodos[k] : nullptr;
    };

    // Estabilizadores, de abajo hacia arriba.
    uint8_t* d_stab[DAG_NIVELES];
    for (int k = 0; k < DAG_NIVELES; ++k) d_stab[k] = nullptr;
    for (int k = n_niv - 1; k >= 0; --k) {
        const long long n = cuenta(k);
        if (n == 0) continue;
        SIM_CHECK(cudaMalloc(&d_stab[k], n));
        simStabKernel<<<simBloques(n), SIM_BLOQUE>>>(nodos(k), (int)n,
                                                     d_stab[k + 1], d_stab[k]);
        SIM_CHECK(cudaGetLastError());
    }
    SIM_CHECK(cudaDeviceSynchronize());

    for (int L = 0; L < n_niv; ++L) {
        SimNivel& s = r->niv[L];
        const long long n = cuenta(L);
        s.n = n;
        if (n == 0) continue;

        // M5: slots NODO.
        {
            thrust::device_ptr<const DagNode> b(nodos(L));
            s.slots_nodo = thrust::transform_reduce(
                thrust::device, b, b + n, SimContarNodo(), (long long)0,
                thrust::plus<long long>());
        }

        // M2 y cota conservadora de M3.
        simM2(nodos(L), n, d_stab[L + 1], 1, &s.d1_canon, &s.d2_canon, &s.deltas_cons);
        simM2(nodos(L), n, d_stab[L + 1], 8, &s.d1_vistas, &s.d2_vistas, nullptr);

        // M1 y M4 comparan hermanos: hacen falta padres en el nivel L-1.
        const long long np = cuenta(L - 1);
        if (L == 0 || np == 0) {
            s.m1[0] = n;
            s.m4[0] = n;
            continue;
        }

        unsigned* d_mejor = nullptr;
        SIM_CHECK(cudaMalloc(&d_mejor, n * sizeof(unsigned)));

        SIM_CHECK(cudaMemset(d_mejor, 0xFF, n * sizeof(unsigned)));
        simM1Kernel<<<simBloques(np), SIM_BLOQUE>>>(nodos(L - 1), (int)np, nodos(L),
                                                    d_stab[L + 1], d_mejor);
        SIM_CHECK(cudaGetLastError());
        SIM_CHECK(cudaDeviceSynchronize());
        s.m1_cero = simContar(d_mejor, n, 0, 0);
        s.m1[1] = simContar(d_mejor, n, 1, 1);
        s.m1[2] = simContar(d_mejor, n, 2, 2);
        s.m1[3] = simContar(d_mejor, n, 3, 3);
        s.m1[4] = simContar(d_mejor, n, 4, 8);
        s.m1[0] = simContar(d_mejor, n, SIM_SIN, SIM_SIN);

        uint32_t* d_rej = nullptr;
        SIM_CHECK(cudaMalloc(&d_rej, (size_t)n * SIM_PALABRAS * sizeof(uint32_t)));
        simRejillaKernel<<<simBloques(n * SIM_PALABRAS), SIM_BLOQUE>>>(
            nodos(L), n, nodos(L + 1), nodos(L + 2), d_rej);
        SIM_CHECK(cudaGetLastError());
        SIM_CHECK(cudaMemset(d_mejor, 0xFF, n * sizeof(unsigned)));
        simM4Kernel<<<simBloques(np), SIM_BLOQUE>>>(nodos(L - 1), (int)np, d_rej, d_mejor);
        SIM_CHECK(cudaGetLastError());
        SIM_CHECK(cudaDeviceSynchronize());
        simMuestra();
        s.m4[1] = simContar(d_mejor, n, 0, 0);
        s.m4[2] = simContar(d_mejor, n, 1, 5);       // <= 1 %
        s.m4[3] = simContar(d_mejor, n, 6, 25);      // <= 5 %
        s.m4[4] = simContar(d_mejor, n, 26, 51);     // <= 10 %
        s.m4[5] = simContar(d_mejor, n, 52, 128);    // <= 25 %
        s.m4[6] = simContar(d_mejor, n, 129, 512);   // > 25 %
        s.m4[0] = simContar(d_mejor, n, SIM_SIN, SIM_SIN);

        SIM_CHECK(cudaFree(d_rej));
        SIM_CHECK(cudaFree(d_mejor));
    }

    for (int k = 0; k < DAG_NIVELES; ++k)
        if (d_stab[k]) SIM_CHECK(cudaFree(d_stab[k]));

    r->pico_medido = g_sim_pico;
    r->segundos = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count();
}

// --------------------------------------------------------------------------
// Totales e impresión.
// --------------------------------------------------------------------------
long long simBytesActual(const SimResultado& r) {
    long long b = 0;
    for (int L = 0; L < DAG_NIVELES; ++L) b += r.niv[L].n * SIM_BYTES_NODO;
    return b;
}

long long simBytesVariable(const SimResultado& r) {
    long long b = 0;
    for (int L = 0; L < DAG_NIVELES; ++L)
        b += r.niv[L].n * 2 + r.niv[L].slots_nodo * 4;
    return b;
}

static long long simOpt(long long n, long long d1, long long d2) {
    return d1 * SIM_BYTES_DELTA1 + d2 * SIM_BYTES_DELTA2 +
           (n - d1 - d2) * SIM_BYTES_NODO;
}

long long simBytesDeltaOptCanon(const SimResultado& r) {
    long long b = 0;
    for (int L = 0; L < DAG_NIVELES; ++L)
        b += simOpt(r.niv[L].n, r.niv[L].d1_canon, r.niv[L].d2_canon);
    return b;
}

long long simBytesDeltaOptVistas(const SimResultado& r) {
    long long b = 0;
    for (int L = 0; L < DAG_NIVELES; ++L)
        b += simOpt(r.niv[L].n, r.niv[L].d1_vistas, r.niv[L].d2_vistas);
    return b;
}

long long simBytesDeltaCons(const SimResultado& r) {
    long long b = 0;
    for (int L = 0; L < DAG_NIVELES; ++L)
        b += (r.niv[L].n - r.niv[L].deltas_cons) * SIM_BYTES_NODO +
             r.niv[L].deltas_cons * SIM_BYTES_DELTA1;
    return b;
}

static const char* simBytes(long long b, char* buf, size_t n) {
    if (b >= (1LL << 30))      snprintf(buf, n, "%.2f GB", b / 1073741824.0);
    else if (b >= (1LL << 20)) snprintf(buf, n, "%.2f MB", b / 1048576.0);
    else if (b >= (1LL << 10)) snprintf(buf, n, "%.2f KB", b / 1024.0);
    else                       snprintf(buf, n, "%lld B", b);
    return buf;
}

void dagImprimirSimilitud(const SimResultado& r) {
    const int n_niv = r.max_nivel > 0 ? r.max_nivel : 0;

    printf("\n=== Similitud aproximada sobre el DAG con espejo (hito 4, solo medición) ===\n");
    printf("Distancia = slots distintos entre dos nodos, mínimo sobre las 8 reflexiones.\n");

    printf("\nM1. Hermano más parecido (hijos NODO distintos de un mismo padre)\n");
    printf(" Nivel |    Únicos | Sin herm. |       d=1 |       d=2 |       d=3 |      d>=4\n");
    printf("-------+-----------+-----------+-----------+-----------+-----------+----------\n");
    long long ceros = 0;
    for (int L = 1; L < n_niv; ++L) {
        const SimNivel& s = r.niv[L];
        if (s.n == 0) continue;
        printf(" %5d | %9lld | %9lld | %9lld | %9lld | %9lld | %9lld\n", L, s.n,
               s.m1[0], s.m1[1], s.m1[2], s.m1[3], s.m1[4]);
        ceros += s.m1_cero;
    }
    if (ceros > 0)
        printf("  AVISO: %lld nodos distintos a distancia 0 (no debería ocurrir).\n", ceros);

    printf("\nM2. Vecino en todo el nivel (d1 = a distancia 1; d2 = a distancia 2 y no 1)\n");
    printf(" Nivel |    Únicos | d1 canón. | d2 canón. | d1 vistas | d2 vistas | Deltas cons.\n");
    printf("-------+-----------+-----------+-----------+-----------+-----------+-------------\n");
    for (int L = 0; L < n_niv; ++L) {
        const SimNivel& s = r.niv[L];
        if (s.n == 0) continue;
        printf(" %5d | %9lld | %9lld | %9lld | %9lld | %9lld | %12lld\n", L, s.n,
               s.d1_canon, s.d2_canon, s.d1_vistas, s.d2_vistas, s.deltas_cons);
    }

    printf("\nM4. Similitud geométrica con el hermano más parecido "
           "(rejilla 8x8x8, %% de celdas distintas)\n");
    printf(" Nivel |    Únicos | Sin herm. |       0 %% |    <=1 %% |    <=5 %% |   <=10 %% |   <=25 %% |    >25 %%\n");
    printf("-------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------\n");
    for (int L = 1; L < n_niv; ++L) {
        const SimNivel& s = r.niv[L];
        if (s.n == 0) continue;
        printf(" %5d | %9lld | %9lld | %9lld | %9lld | %9lld | %9lld | %9lld | %9lld\n",
               L, s.n, s.m4[0], s.m4[1], s.m4[2], s.m4[3], s.m4[4], s.m4[5], s.m4[6]);
    }

    const long long actual = simBytesActual(r);
    char b1[32], b2[32], b3[32], b4[32], b5[32], b6[32];
    auto pct = [&](long long b) { return actual > 0 ? 100.0 * b / actual : 0.0; };
    printf("\nTamaños estimados (nodo completo %d B; delta de 1 slot %d B; de 2 slots %d B)\n",
           SIM_BYTES_NODO, SIM_BYTES_DELTA1, SIM_BYTES_DELTA2);
    printf("  DAG actual (únicos x 32 B)                : %12s\n", simBytes(actual, b1, sizeof(b1)));
    printf("  M5 nodos de tamaño variable, sin deltas    : %12s (%.1f %%)\n",
           simBytes(simBytesVariable(r), b2, sizeof(b2)), pct(simBytesVariable(r)));
    printf("  M3 delta, cota conservadora (1 posición)   : %12s (%.1f %%)\n",
           simBytes(simBytesDeltaCons(r), b3, sizeof(b3)), pct(simBytesDeltaCons(r)));
    printf("  M3 delta, cota optimista (formas canónicas): %12s (%.1f %%)\n",
           simBytes(simBytesDeltaOptCanon(r), b4, sizeof(b4)), pct(simBytesDeltaOptCanon(r)));
    printf("  M3 delta, cota optimista (8 vistas)        : %12s (%.1f %%)\n",
           simBytes(simBytesDeltaOptVistas(r), b5, sizeof(b5)), pct(simBytesDeltaOptVistas(r)));
    printf("  Costo de la medición: %.3f s, VRAM adicional máxima %s\n",
           r.segundos, simBytes((long long)r.pico_medido, b6, sizeof(b6)));
}
