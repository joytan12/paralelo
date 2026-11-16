// ==========================================================================
// DAG global incremental (hito 2a). Este archivo se incluye al final de
// dag.cu: comparte sus kernels y utilidades (dalloc, canonizarKernel, ...).
//
// Se agregan subárboles de a uno. Por nivel k se guardan los nodos únicos en
// un arreglo de índices estables y solo-agregar (un nodo guardado nunca
// cambia de índice, porque los niveles superiores de otros subárboles ya lo
// referencian), su estabilizador y una permutación de los índices ordenada
// por contenido para buscar. Insertar un lote: armar padres con índices
// globales de los hijos, canonicalizar con los estabilizadores globales,
// sort + unique del lote, búsqueda binaria contra la permutación, agregar los
// que faltan y fusionar la permutación (merge).
//
// La canonicalización se hace siempre con índices globales ya asignados: el
// orden lexicográfico depende de ellos, así que nunca se remapean después.
// ==========================================================================

struct DagGlobal {
    bool       espejo;
    long long  cap_inicial;
    DagNode*   nodos[DAG_NIVELES];
    uint8_t*   estab[DAG_NIVELES];
    uint32_t*  perm[DAG_NIVELES];
    long long  cuenta[DAG_NIVELES];
    long long  cap[DAG_NIVELES];
};

DagGlobal* dagGlobalCrear(bool espejo, long long cap_inicial) {
    DagGlobal* g = new DagGlobal();
    g->espejo = espejo;
    g->cap_inicial = cap_inicial < 1 ? 1 : cap_inicial;
    return g;
}

void dagGlobalLiberar(DagGlobal* g) {
    if (!g) return;
    for (int l = 0; l < DAG_NIVELES; ++l) {
        dfree(g->nodos[l], g->cap[l]);
        dfree(g->estab[l], g->cap[l]);
        dfree(g->perm[l], g->cuenta[l]);
    }
    delete g;
}

long long dagGlobalCuenta(const DagGlobal* g, int nivel) {
    return g->cuenta[nivel];
}

// Reserva (con contabilidad) las referencias (clave, slot) de n subárboles,
// inicializadas en 0 = VACÍO. dagGlobalCompletar las libera.
void dagReservarRefs(int n, unsigned long long** d_claves, uint32_t** d_slots) {
    *d_claves = dalloc<unsigned long long>(n);
    *d_slots = dalloc<uint32_t>(n);
    CUDA_CHECK(cudaMemset(*d_claves, 0, (size_t)(n > 0 ? n : 1) *
                          sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(*d_slots, 0, (size_t)(n > 0 ? n : 1) *
                          sizeof(uint32_t)));
}

// Crece nodos/estab del nivel k sin pasar por el host (copia device a device).
static void asegurarCapacidad(DagGlobal* g, int k, long long necesario) {
    if (necesario <= g->cap[k]) return;
    long long nueva = g->cap[k] * 2;
    if (nueva < necesario) nueva = necesario;
    if (nueva < g->cap_inicial) nueva = g->cap_inicial;
    if (nueva > (long long)DAG_MASK_IDX + 1) nueva = (long long)DAG_MASK_IDX + 1;
    DagNode* n2 = dalloc<DagNode>((size_t)nueva);
    uint8_t* e2 = dalloc<uint8_t>((size_t)nueva);
    if (g->cuenta[k] > 0) {
        CUDA_CHECK(cudaMemcpy(n2, g->nodos[k],
                              (size_t)g->cuenta[k] * sizeof(DagNode),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(e2, g->estab[k], (size_t)g->cuenta[k],
                              cudaMemcpyDeviceToDevice));
    }
    dfree(g->nodos[k], g->cap[k]);
    dfree(g->estab[k], g->cap[k]);
    g->nodos[k] = n2;
    g->estab[k] = e2;
    g->cap[k] = nueva;
}

static __device__ __forceinline__ int cmpNodo(const DagNode& a,
                                              const DagNode& b) {
    for (int o = 0; o < 8; ++o) {
        if (a.child[o] != b.child[o]) return a.child[o] < b.child[o] ? -1 : 1;
    }
    return 0;
}

// Busca cada nodo único del lote (lower_bound) en la permutación global.
__global__ void buscarGlobalKernel(const DagNode* lote, int n_lote,
                                   const DagNode* nodos, const uint32_t* perm,
                                   long long cuenta, uint32_t* gid,
                                   int* nuevo) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_lote) return;
    long long lo = 0, hi = cuenta;
    while (lo < hi) {
        const long long mid = lo + (hi - lo) / 2;
        if (cmpNodo(nodos[perm[mid]], lote[j]) < 0) lo = mid + 1;
        else hi = mid;
    }
    if (lo < cuenta && cmpNodo(nodos[perm[lo]], lote[j]) == 0) {
        gid[j] = perm[lo];
        nuevo[j] = 0;
    } else {
        nuevo[j] = 1;
    }
}

// Agrega los nodos nuevos con índices consecutivos. Como el lote está
// ordenado por contenido, nl (los índices nuevos) sale ordenado por contenido.
__global__ void asignarGlobalKernel(const DagNode* lote,
                                    const uint8_t* lote_estab, const int* nuevo,
                                    const int* rango, int n_lote,
                                    DagNode* nodos, uint8_t* estab,
                                    long long base, uint32_t* gid,
                                    uint32_t* nl) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_lote || !nuevo[j]) return;
    const uint32_t idx = (uint32_t)(base + rango[j]);
    gid[j] = idx;
    nodos[idx] = lote[j];
    estab[idx] = lote_estab[j];
    nl[rango[j]] = idx;
}

struct MenorIdx {
    const DagNode* nodos;
    __host__ __device__ bool operator()(uint32_t a, uint32_t b) const {
        return NodoMenor()(nodos[a], nodos[b]);
    }
};

__global__ void referenciasGlobalKernel(const uint32_t* ids, const int* u,
                                        const uint32_t* masc,
                                        const uint32_t* gid, int n,
                                        uint32_t* nuevo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t p = ids[i];
    nuevo[p] = dagSlot(DAG_NODO, masc[p], gid[u[i] - 1]);
}

struct NoCero {
    __host__ __device__ bool operator()(uint32_t v) const { return v != 0; }
};

// Procesa las entradas (clave, slot) del nivel L: arma los padres, los
// canonicaliza, los inserta en el nivel L-1 global y devuelve los padres como
// entradas del nivel L-1 (claves ordenadas). Toma posesión de claves/slots.
void dagGlobalSubirNivel(DagGlobal* g, int L, unsigned long long* claves,
                         uint32_t* slots, int E,
                         unsigned long long** out_claves, uint32_t** out_slots,
                         int* n_out) {
    const int k = L - 1;

    // 1) Agrupar por padre.
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

    // 2) Canonicalizar con los estabilizadores globales del nivel L.
    DagNode*  canon = dalloc<DagNode>(n_par);
    uint32_t* masc  = dalloc<uint32_t>(n_par);
    uint8_t*  estab_padre = dalloc<uint8_t>(n_par);
    canonizarKernel<<<bloques(n_par), THREADS>>>(
        crudo, n_par, g->espejo, g->estab[L], canon, masc, estab_padre);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    dfree(crudo, n_par);

    // 3) sort + unique del lote (igualdad exacta de los 8 uint32).
    uint32_t* ids = dalloc<uint32_t>(n_par);
    thrust::sequence(thrust::device, ids, ids + n_par);
    thrust::sort_by_key(thrust::device, canon, canon + n_par, ids, NodoMenor());
    int* cab2 = dalloc<int>(n_par);
    int* u    = dalloc<int>(n_par);
    cabezasUnicoKernel<<<bloques(n_par), THREADS>>>(canon, n_par, cab2);
    CUDA_CHECK(cudaGetLastError());
    thrust::inclusive_scan(thrust::device, cab2, cab2 + n_par, u);
    const int n_lote = leerContador(u + (n_par - 1));

    DagNode* lote = dalloc<DagNode>(n_lote);
    thrust::copy_if(thrust::device, canon, canon + n_par, cab2, lote, EsUno());
    uint8_t* lote_estab = dalloc<uint8_t>(n_lote);
    estabilizadorUnicoKernel<<<bloques(n_par), THREADS>>>(
        ids, cab2, u, estab_padre, n_par, lote_estab);
    CUDA_CHECK(cudaGetLastError());
    dfree(canon, n_par);
    dfree(cab2, n_par);
    dfree(estab_padre, n_par);

    // 4) Buscar en lo ya guardado y agregar lo que falta.
    uint32_t* gid   = dalloc<uint32_t>(n_lote);
    int*      nuevo = dalloc<int>(n_lote);
    int*      rango = dalloc<int>(n_lote);
    buscarGlobalKernel<<<bloques(n_lote), THREADS>>>(
        lote, n_lote, g->nodos[k], g->perm[k], g->cuenta[k], gid, nuevo);
    CUDA_CHECK(cudaGetLastError());
    thrust::exclusive_scan(thrust::device, nuevo, nuevo + n_lote, rango);
    const int n_nuevos = leerContador(rango + (n_lote - 1)) +
                         leerContador(nuevo + (n_lote - 1));

    if (n_nuevos > 0) {
        if (g->cuenta[k] + n_nuevos > (long long)DAG_MASK_IDX + 1) {
            fprintf(stderr, "dagGlobal: el nivel %d superaría los %d bits de índice\n",
                    k, DAG_BITS_IDX);
            exit(EXIT_FAILURE);
        }
        asegurarCapacidad(g, k, g->cuenta[k] + n_nuevos);
        uint32_t* nl = dalloc<uint32_t>(n_nuevos);
        asignarGlobalKernel<<<bloques(n_lote), THREADS>>>(
            lote, lote_estab, nuevo, rango, n_lote, g->nodos[k], g->estab[k],
            g->cuenta[k], gid, nl);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 5) Fusionar la permutación ordenada por contenido.
        uint32_t* perm2 = dalloc<uint32_t>((size_t)(g->cuenta[k] + n_nuevos));
        thrust::merge(thrust::device, g->perm[k], g->perm[k] + g->cuenta[k],
                      nl, nl + n_nuevos, perm2, MenorIdx{g->nodos[k]});
        dfree(g->perm[k], g->cuenta[k]);
        g->perm[k] = perm2;
        g->cuenta[k] += n_nuevos;
        dfree(nl, n_nuevos);
    }
    dfree(lote, n_lote);
    dfree(lote_estab, n_lote);
    dfree(nuevo, n_lote);
    dfree(rango, n_lote);

    // 6) Cada padre queda como (idx global, máscara).
    uint32_t* refs = dalloc<uint32_t>(n_par);
    referenciasGlobalKernel<<<bloques(n_par), THREADS>>>(
        ids, u, masc, gid, n_par, refs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    dfree(ids, n_par);
    dfree(u, n_par);
    dfree(masc, n_par);
    dfree(gid, n_lote);

    *out_claves = clave_padre;
    *out_slots = refs;
    *n_out = n_par;
}

__global__ void hojasAEntradasKernel(const Cube* cubos, int n,
                                     unsigned long long* claves,
                                     uint32_t* slots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    claves[i] = cubos[i].path_code;
    slots[i]  = slotDeHoja(cubos[i].state);
}

void dagGlobalInsertarSubarbol(DagGlobal* g, const Cube* d_hojas, int n,
                               int nivel_hojas, int nivel_raiz,
                               unsigned long long* d_clave_out,
                               uint32_t* d_slot_out) {
    unsigned long long* claves = dalloc<unsigned long long>(n);
    uint32_t* slots = dalloc<uint32_t>(n);
    hojasAEntradasKernel<<<bloques(n), THREADS>>>(d_hojas, n, claves, slots);
    CUDA_CHECK(cudaGetLastError());

    int cuantas = n;
    for (int L = nivel_hojas; L > nivel_raiz; --L) {
        unsigned long long* c2 = nullptr;
        uint32_t* s2 = nullptr;
        int n2 = 0;
        dagGlobalSubirNivel(g, L, claves, slots, cuantas, &c2, &s2, &n2);
        claves = c2; slots = s2; cuantas = n2;
    }
    if (cuantas != 1) {
        fprintf(stderr, "dagGlobalInsertarSubarbol: se esperaba 1 raíz de "
                "subárbol y hay %d\n", cuantas);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(d_clave_out, claves, sizeof(unsigned long long),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot_out, slots, sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));
    dfree(claves, cuantas);
    dfree(slots, cuantas);
}

void dagGlobalCompletar(DagGlobal* g, int nivel_refs,
                        unsigned long long* d_claves, uint32_t* d_slots, int n,
                        int max_nivel, Dag* dag) {
    // Los subárboles vacíos llegan con slot 0 (VACÍO): se descartan. Los
    // buffers filtrados se reservan con tamaño exacto.
    int cuantas = (int)thrust::count_if(thrust::device, d_slots, d_slots + n,
                                        NoCero());
    unsigned long long* claves = dalloc<unsigned long long>(cuantas);
    uint32_t* slots = dalloc<uint32_t>(cuantas);
    thrust::copy_if(thrust::device, d_claves, d_claves + n, d_slots, claves,
                    NoCero());
    thrust::copy_if(thrust::device, d_slots, d_slots + n, d_slots, slots,
                    NoCero());
    dfree(d_claves, n);
    dfree(d_slots, n);

    *dag = Dag{};
    dag->espejo = g->espejo;
    dag->max_nivel = max_nivel;
    dag->d_raiz = dalloc<uint32_t>(1);
    CUDA_CHECK(cudaMemset(dag->d_raiz, 0, sizeof(uint32_t)));   // VACÍO

    // Se suben los niveles nivel_refs..1; cada llamada devuelve buffers de
    // tamaño igual a la cantidad de padres.
    for (int L = nivel_refs; L >= 1 && cuantas > 0; --L) {
        unsigned long long* c2 = nullptr;
        uint32_t* s2 = nullptr;
        int n2 = 0;
        dagGlobalSubirNivel(g, L, claves, slots, cuantas, &c2, &s2, &n2);
        claves = c2; slots = s2; cuantas = n2;
    }
    if (cuantas > 1) {
        fprintf(stderr, "dagGlobalCompletar: se esperaba 1 raíz y hay %d\n",
                cuantas);
        exit(EXIT_FAILURE);
    }
    if (cuantas == 1)
        CUDA_CHECK(cudaMemcpy(dag->d_raiz, slots, sizeof(uint32_t),
                              cudaMemcpyDeviceToDevice));
    dfree(claves, cuantas);
    dfree(slots, cuantas);

    // El DAG se queda con los nodos; la permutación y los estabilizadores
    // solo hacen falta mientras se agregan subárboles.
    for (int l = 0; l < DAG_NIVELES; ++l) {
        dag->d_nodos[l] = g->nodos[l];
        dag->n_unicos[l] = g->cuenta[l];
        g->nodos[l] = nullptr;
        dfree(g->perm[l], g->cuenta[l]);
        dfree(g->estab[l], g->cap[l]);
        g->perm[l] = nullptr;
        g->estab[l] = nullptr;
        g->cap[l] = 0;
        g->cuenta[l] = 0;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// --------------------------------------------------------------------------
// Conteos desde el DAG, sin expandir: se propaga de arriba hacia abajo la
// multiplicidad (cuántas veces aparece cada nodo único en el árbol lógico).
// Una hoja aporta la multiplicidad de su nodo; la reflexión no cambia el
// tipo ni el nivel de las hojas, así que no interviene.
//
// cont: [0] INSIDE, [1] BORDER, [2+l] hojas de nivel l,
//       [2+DAG_NIVELES+l] nodos lógicos de nivel l.
// --------------------------------------------------------------------------
__global__ void multRaizKernel(const uint32_t* raiz, unsigned long long* mult0,
                               unsigned long long* cont) {
    const uint32_t s = raiz[0];
    const uint32_t t = dagTipo(s);
    if (t == DAG_NODO) {
        if (mult0 != nullptr) mult0[dagIdx(s)] = 1ULL;
    } else if (esHoja(t)) {
        cont[t == DAG_HOJA_INSIDE ? 0 : 1] += 1ULL;
        cont[2] += 1ULL;
    }
}

__global__ void multNivelKernel(const DagNode* nodos, int n,
                                const unsigned long long* mult_k,
                                unsigned long long* mult_k1, int k,
                                unsigned long long* cont) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long m = mult_k[i];
    if (m == 0) return;
    atomicAdd(&cont[2 + DAG_NIVELES + k], m);
    unsigned long long ci = 0, cb = 0;
    for (int o = 0; o < 8; ++o) {
        const uint32_t s = nodos[i].child[o];
        const uint32_t t = dagTipo(s);
        if (t == DAG_NODO) {
            if (mult_k1 != nullptr) atomicAdd(&mult_k1[dagIdx(s)], m);
        } else if (t == DAG_HOJA_INSIDE) {
            ci += m;
        } else if (t == DAG_HOJA_BORDER) {
            cb += m;
        }
    }
    if (ci) atomicAdd(&cont[0], ci);
    if (cb) atomicAdd(&cont[1], cb);
    if (ci + cb) atomicAdd(&cont[2 + k + 1], ci + cb);
}

void dagContarDesdeDag(Dag* dag, long long* n_inside, long long* n_border) {
    const int ncont = 2 + 2 * DAG_NIVELES;
    unsigned long long* d_cont = dalloc<unsigned long long>(ncont);
    CUDA_CHECK(cudaMemset(d_cont, 0, ncont * sizeof(unsigned long long)));

    unsigned long long* mult[DAG_NIVELES] = {};
    const int niveles_internos = dag->max_nivel > 0 ? dag->max_nivel : 0;
    for (int k = 0; k < niveles_internos; ++k) {
        mult[k] = dalloc<unsigned long long>((size_t)dag->n_unicos[k]);
        CUDA_CHECK(cudaMemset(mult[k], 0,
                              (size_t)(dag->n_unicos[k] > 0 ? dag->n_unicos[k] : 1) *
                              sizeof(unsigned long long)));
    }
    multRaizKernel<<<1, 1>>>(dag->d_raiz, mult[0], d_cont);
    CUDA_CHECK(cudaGetLastError());
    for (int k = 0; k < niveles_internos; ++k) {
        const int n = (int)dag->n_unicos[k];
        if (n == 0) continue;
        multNivelKernel<<<bloques(n), THREADS>>>(
            dag->d_nodos[k], n, mult[k],
            (k + 1 < niveles_internos) ? mult[k + 1] : nullptr, k, d_cont);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long cont[2 + 2 * DAG_NIVELES];
    CUDA_CHECK(cudaMemcpy(cont, d_cont, sizeof(cont), cudaMemcpyDeviceToHost));
    for (int k = 0; k < niveles_internos; ++k)
        dfree(mult[k], (size_t)dag->n_unicos[k]);
    dfree(d_cont, ncont);

    *n_inside = (long long)cont[0];
    *n_border = (long long)cont[1];
    for (int l = 0; l < DAG_NIVELES; ++l) {
        dag->n_hojas[l]   = (long long)cont[2 + l];
        dag->n_logicos[l] = (long long)cont[2 + DAG_NIVELES + l];
    }
}

// --------------------------------------------------------------------------
// Comparación de dos DAG: mismas hojas (path, nivel, estado) y mismo número
// de nodos únicos por nivel. Los índices y las máscaras pueden diferir.
// --------------------------------------------------------------------------
bool dagCompararDags(const Dag& a, const Dag& b, bool verboso) {
    long long na = 0, nb = 0;
    RegistroHoja* ra = expandirDag(a, &na);
    RegistroHoja* rb = expandirDag(b, &nb);
    bool ok = (na == nb);
    unsigned long long difs = 0;
    if (ok && na > 0) {
        unsigned long long* d_difs = dalloc<unsigned long long>(1);
        CUDA_CHECK(cudaMemset(d_difs, 0, sizeof(unsigned long long)));
        compararRegistrosKernel<<<bloques(na), THREADS>>>(ra, rb, (int)na, d_difs);
        CUDA_CHECK(cudaGetLastError());
        difs = leerContador(d_difs);
        dfree(d_difs, 1);
        ok = (difs == 0);
    }
    dfree(ra, na > 0 ? na : 1);
    dfree(rb, nb > 0 ? nb : 1);

    bool niveles_ok = true;
    for (int l = 0; l < DAG_NIVELES; ++l) {
        if (a.n_unicos[l] != b.n_unicos[l]) {
            niveles_ok = false;
            if (verboso)
                printf("    nivel %d: %lld únicos vs %lld únicos\n", l,
                       a.n_unicos[l], b.n_unicos[l]);
        }
    }
    if (verboso) {
        printf("  Comparación de DAG: %lld vs %lld hojas expandidas, %llu "
               "diferencias -> %s; únicos por nivel -> %s\n", na, nb, difs,
               ok ? "OK" : "FALLO", niveles_ok ? "OK" : "FALLO");
    }
    return ok && niveles_ok;
}

// --------------------------------------------------------------------------
// Verificación por bloques: se expande el DAG subárbol por subárbol (cada
// bloque nace de una celda del nivel `nivel_bloque`), se consulta cada hoja
// con dagLookup y, si hay `regenerar`, se rehace el subárbol desde la
// malla/clasificador y se compara hoja a hoja. Cada bloque se libera antes de
// pasar al siguiente, así que nunca existe el arreglo completo de hojas.
// --------------------------------------------------------------------------
__global__ void consultarRegistrosKernel(DagView v, const RegistroHoja* reg,
                                         int n, unsigned long long* malas) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int nivel = reg[i].nivel;
    int x = 0, y = 0, z = 0;
    for (int k = 0; k < nivel; ++k) {
        const int o = (int)((reg[i].path >> (3 * (nivel - 1 - k))) & 7ULL);
        x = (x << 1) | (o & 1);
        y = (y << 1) | ((o >> 1) & 1);
        z = (z << 1) | ((o >> 2) & 1);
    }
    const DagHit h = dagLookup(v, x, y, z, nivel);
    if (h.tipo != (uint32_t)reg[i].estado || h.nivel != nivel)
        atomicAdd(malas, 1ULL);
}

__global__ void buscarRaizKernel(const Cube* raices, int n,
                                 const unsigned long long* path, int* idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (raices[i].path_code == path[0]) idx[0] = i;
}

static unsigned long long consultarBloque(const Dag& dag,
                                          const RegistroHoja* reg, long long n) {
    if (n <= 0) return 0;
    unsigned long long* d_malas = dalloc<unsigned long long>(1);
    CUDA_CHECK(cudaMemset(d_malas, 0, sizeof(unsigned long long)));
    consultarRegistrosKernel<<<bloques(n), THREADS>>>(
        dagView(dag), reg, (int)n, d_malas);
    CUDA_CHECK(cudaGetLastError());
    const unsigned long long m = leerContador(d_malas);
    dfree(d_malas, 1);
    return m;
}

bool dagVerificarPorBloques(const Dag& dag, int nivel_bloque,
                            const Cube* d_raices, int n_raices,
                            const DagRegenerarFn& regenerar, bool verboso,
                            long long* n_hojas_total) {
    unsigned long long* fpath = dalloc<unsigned long long>(1);
    uint32_t* fslot = dalloc<uint32_t>(1);
    iniciarFrenteKernel<<<1, 1>>>(dag.d_raiz, fpath, fslot);
    CUDA_CHECK(cudaGetLastError());

    long long n_tempranas = 0, n_frente = 0;
    unsigned long long* bpath = nullptr;
    uint32_t* bslot = nullptr;
    RegistroHoja* tempranas = expandirDesde(
        dag, 0, fpath, fslot, 1, nivel_bloque, &n_tempranas, &bpath, &bslot,
        &n_frente);

    long long total = n_tempranas;
    unsigned long long malas = consultarBloque(dag, tempranas, n_tempranas);
    unsigned long long difs_regen = 0;
    dfree(tempranas, n_tempranas > 0 ? n_tempranas : 1);

    for (long long i = 0; i < n_frente; ++i) {
        unsigned long long* ipath = dalloc<unsigned long long>(1);
        uint32_t* islot = dalloc<uint32_t>(1);
        CUDA_CHECK(cudaMemcpy(ipath, bpath + i, sizeof(unsigned long long),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(islot, bslot + i, sizeof(uint32_t),
                              cudaMemcpyDeviceToDevice));

        // Para regenerar hace falta la raíz del subárbol: se busca por path.
        int raiz_idx = -1;
        if (regenerar) {
            int* d_idx = dalloc<int>(1);
            const int menos1 = -1;
            CUDA_CHECK(cudaMemcpy(d_idx, &menos1, sizeof(int),
                                  cudaMemcpyHostToDevice));
            buscarRaizKernel<<<bloques(n_raices), THREADS>>>(
                d_raices, n_raices, ipath, d_idx);
            CUDA_CHECK(cudaGetLastError());
            raiz_idx = leerContador(d_idx);
            dfree(d_idx, 1);
        }

        long long n_blk = 0;
        RegistroHoja* blk = expandirDesde(dag, nivel_bloque, ipath, islot, 1,
                                          DAG_NIVELES + 1, &n_blk);
        total += n_blk;
        malas += consultarBloque(dag, blk, n_blk);

        if (regenerar) {
            if (raiz_idx < 0) {
                ++difs_regen;                    // no hay raíz para este bloque
            } else {
                Cube* d_h = nullptr;
                int n_h = 0;
                regenerar(d_raices + raiz_idx, &d_h, &n_h);
                if ((long long)n_h != n_blk) {
                    ++difs_regen;
                } else if (n_h > 0) {
                    RegistroHoja* orig = dalloc<RegistroHoja>(n_h);
                    registrosOriginalesKernel<<<bloques(n_h), THREADS>>>(
                        d_h, n_h, orig);
                    CUDA_CHECK(cudaGetLastError());
                    if (n_h > 1)
                        thrust::sort(thrust::device, orig, orig + n_h,
                                     RegistroMenor());
                    unsigned long long* d_difs = dalloc<unsigned long long>(1);
                    CUDA_CHECK(cudaMemset(d_difs, 0, sizeof(unsigned long long)));
                    compararRegistrosKernel<<<bloques(n_h), THREADS>>>(
                        blk, orig, n_h, d_difs);
                    CUDA_CHECK(cudaGetLastError());
                    difs_regen += leerContador(d_difs);
                    dfree(d_difs, 1);
                    dfree(orig, n_h);
                }
                if (d_h) CUDA_CHECK(cudaFree(d_h));
            }
        }
        dfree(blk, n_blk > 0 ? n_blk : 1);
    }
    dfree(bpath, n_frente > 0 ? n_frente : 1);
    dfree(bslot, n_frente > 0 ? n_frente : 1);

    if (n_hojas_total) *n_hojas_total = total;
    const bool ok = (malas == 0 && difs_regen == 0);
    if (verboso) {
        printf("  Verificación por bloques (%s): %lld bloques, %lld hojas, "
               "dagLookup incorrectos: %llu",
               dag.espejo ? "con espejo" : "sin espejo", n_frente, total, malas);
        if (regenerar)
            printf(", diferencias contra regeneración: %llu", difs_regen);
        printf(" -> %s\n", ok ? "OK" : "FALLO");
    }
    return ok;
}

// Entradas (clave, slot) de un arreglo de hojas del mismo nivel.
void dagEntradasDeHojas(const Cube* d_hojas, int n,
                        unsigned long long* d_claves, uint32_t* d_slots) {
    if (n <= 0) return;
    hojasAEntradasKernel<<<bloques(n), THREADS>>>(d_hojas, n, d_claves, d_slots);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==========================================================================
// Hito 2b: subárboles adaptativos (hojas en varios niveles).
// ==========================================================================

// Comprime un subárbol adaptativo contra el DAG global. Las hojas de d_hojas
// pueden estar en cualquier nivel > nivel_raiz (o ser la propia raíz del
// subárbol, un único cubo de nivel nivel_raiz). Se sube nivel por nivel: en
// cada uno las entradas son los padres que vienen de abajo más las hojas de
// ese nivel. La referencia queda en d_clave_out / d_slot_out.
void dagGlobalInsertarSubarbolGeneral(DagGlobal* g, const Cube* d_hojas, int n,
                                      int nivel_raiz,
                                      unsigned long long* d_clave_out,
                                      uint32_t* d_slot_out) {
    // Hojas por nivel (22 enteros al host).
    int* d_hist = dalloc<int>(DAG_NIVELES + 1);
    CUDA_CHECK(cudaMemset(d_hist, 0, (DAG_NIVELES + 1) * sizeof(int)));
    histNivelesKernel<<<bloques(n), THREADS>>>(d_hojas, n, d_hist);
    CUDA_CHECK(cudaGetLastError());
    int hist[DAG_NIVELES + 1];
    CUDA_CHECK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));
    dfree(d_hist, DAG_NIVELES + 1);
    if (hist[DAG_NIVELES] > 0) {
        fprintf(stderr, "dagGlobalInsertarSubarbolGeneral: nivel fuera de rango\n");
        exit(EXIT_FAILURE);
    }
    int lmax = -1;
    for (int l = 0; l < DAG_NIVELES; ++l)
        if (hist[l] > 0) lmax = l;

    // Caso degenerado: el subárbol es una sola hoja (su propia raíz).
    if (lmax <= nivel_raiz) {
        if (n != 1) {
            fprintf(stderr, "dagGlobalInsertarSubarbolGeneral: %d hojas en el "
                    "nivel de la raíz del subárbol\n", n);
            exit(EXIT_FAILURE);
        }
        dagEntradasDeHojas(d_hojas, 1, d_clave_out, d_slot_out);
        return;
    }

    unsigned long long* carry_claves = nullptr;
    uint32_t* carry_slots = nullptr;
    int n_carry = 0;

    for (int L = lmax; L > nivel_raiz; --L) {
        const int n_hojas = hist[L];
        const int E = n_carry + n_hojas;
        unsigned long long* claves = dalloc<unsigned long long>(E);
        uint32_t* slots = dalloc<uint32_t>(E);
        if (n_carry > 0) {
            CUDA_CHECK(cudaMemcpy(claves, carry_claves,
                                  (size_t)n_carry * sizeof(unsigned long long),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(slots, carry_slots,
                                  (size_t)n_carry * sizeof(uint32_t),
                                  cudaMemcpyDeviceToDevice));
            dfree(carry_claves, n_carry);
            dfree(carry_slots, n_carry);
            carry_claves = nullptr;
            carry_slots = nullptr;
        }
        if (n_hojas > 0) {
            int* sel = dalloc<int>(n_hojas);
            EsNivel pred{d_hojas, L};
            thrust::copy_if(thrust::device, thrust::counting_iterator<int>(0),
                            thrust::counting_iterator<int>(n), sel, pred);
            recogerHojasKernel<<<bloques(n_hojas), THREADS>>>(
                d_hojas, sel, n_hojas, claves + n_carry, slots + n_carry);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            dfree(sel, n_hojas);
        }
        unsigned long long* c2 = nullptr;
        uint32_t* s2 = nullptr;
        int n2 = 0;
        dagGlobalSubirNivel(g, L, claves, slots, E, &c2, &s2, &n2);
        carry_claves = c2; carry_slots = s2; n_carry = n2;
    }
    if (n_carry != 1) {
        fprintf(stderr, "dagGlobalInsertarSubarbolGeneral: se esperaba 1 raíz de "
                "subárbol y hay %d\n", n_carry);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(d_clave_out, carry_claves, sizeof(unsigned long long),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot_out, carry_slots, sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));
    dfree(carry_claves, n_carry);
    dfree(carry_slots, n_carry);
}

// Descarta una pasada: libera el DAG global y las referencias (sin construir
// el DAG final).
void dagGlobalDescartar(DagGlobal* g, unsigned long long* d_claves,
                        uint32_t* d_slots, int n) {
    dfree(d_claves, n);
    dfree(d_slots, n);
    dagGlobalLiberar(g);
}

// --------------------------------------------------------------------------
// Verificación por bloques en adaptativo: balance 2:1 global y consultas.
// Cada bloque (celda del nivel nivel_bloque) se expande, se consultan los 26
// vecinos de cada hoja con dagLookup sobre el DAG COMPLETO y se libera antes
// de pasar al siguiente. Es independiente del chequeo de cáscaras.
// --------------------------------------------------------------------------
bool dagVerificarBloquesAdaptivo(const Dag& dag, int nivel_bloque, bool verboso,
                                 long long* n_hojas_total,
                                 long long* violaciones_out) {
    unsigned long long* fpath = dalloc<unsigned long long>(1);
    uint32_t* fslot = dalloc<uint32_t>(1);
    iniciarFrenteKernel<<<1, 1>>>(dag.d_raiz, fpath, fslot);
    CUDA_CHECK(cudaGetLastError());

    long long n_tempranas = 0, n_frente = 0;
    unsigned long long* bpath = nullptr;
    uint32_t* bslot = nullptr;
    RegistroHoja* tempranas = expandirDesde(
        dag, 0, fpath, fslot, 1, nivel_bloque, &n_tempranas, &bpath, &bslot,
        &n_frente);

    unsigned long long* d_viol = dalloc<unsigned long long>(1);
    CUDA_CHECK(cudaMemset(d_viol, 0, sizeof(unsigned long long)));
    long long total = n_tempranas;
    unsigned long long malas = consultarBloque(dag, tempranas, n_tempranas);
    if (n_tempranas > 0) {
        balanceKernel<<<bloques(n_tempranas), THREADS>>>(
            dagView(dag), tempranas, (int)n_tempranas, d_viol);
        CUDA_CHECK(cudaGetLastError());
    }
    dfree(tempranas, n_tempranas > 0 ? n_tempranas : 1);

    for (long long i = 0; i < n_frente; ++i) {
        unsigned long long* ipath = dalloc<unsigned long long>(1);
        uint32_t* islot = dalloc<uint32_t>(1);
        CUDA_CHECK(cudaMemcpy(ipath, bpath + i, sizeof(unsigned long long),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(islot, bslot + i, sizeof(uint32_t),
                              cudaMemcpyDeviceToDevice));
        long long n_blk = 0;
        RegistroHoja* blk = expandirDesde(dag, nivel_bloque, ipath, islot, 1,
                                          DAG_NIVELES + 1, &n_blk);
        total += n_blk;
        malas += consultarBloque(dag, blk, n_blk);
        if (n_blk > 0) {
            balanceKernel<<<bloques(n_blk), THREADS>>>(
                dagView(dag), blk, (int)n_blk, d_viol);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        dfree(blk, n_blk > 0 ? n_blk : 1);
    }
    const unsigned long long viol = leerContador(d_viol);
    dfree(d_viol, 1);
    dfree(bpath, n_frente > 0 ? n_frente : 1);
    dfree(bslot, n_frente > 0 ? n_frente : 1);

    if (n_hojas_total) *n_hojas_total = total;
    if (violaciones_out) *violaciones_out = (long long)viol;
    const bool ok = (viol == 0 && malas == 0);
    if (verboso) {
        printf("  Balance 2:1 global por bloques (%s): %lld bloques, %lld hojas, "
               "%llu violaciones, dagLookup incorrectos: %llu -> %s\n",
               dag.espejo ? "con espejo" : "sin espejo", n_frente, total, viol,
               malas, ok ? "OK" : "FALLO");
    }
    return ok;
}

// ==========================================================================
// Hito 3: comparación por cobertura (para la limitación de la fase A).
//
// Cuando la fase A subdivide cubos INSIDE que el monolítico dejaría gruesos,
// el DAG por subárboles es un octree válido pero más refinado en el interior.
// Esta comparación comprueba exactamente eso:
//   - las hojas BORDER son idénticas (mismo path, nivel y estado);
//   - toda hoja INSIDE de `sub` está contenida en una hoja INSIDE de `ref`
//     (ella misma o un ancestro);
//   - el volumen INSIDE total es el mismo.
// Si las tres se cumplen, ambos árboles cubren exactamente la misma región
// INSIDE/BORDER.
// ==========================================================================
struct EsEstadoReg {
    int estado;
    __host__ __device__ bool operator()(const RegistroHoja& r) const {
        return r.estado == estado;
    }
};

__global__ void clavesDeRegistrosKernel(const RegistroHoja* reg, int n,
                                        unsigned long long* claves) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    claves[i] = reg[i].path;
}

// ¿Alguna clave de `claves` (ordenadas) es el path de la hoja o de un ancestro?
__global__ void contenidaPorKernel(const unsigned long long* claves, int n_claves,
                                   const RegistroHoja* reg, int n,
                                   unsigned long long* no_contenidas) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int nivel = reg[i].nivel;
    bool hallada = false;
    for (int l = nivel; l >= 0 && !hallada; --l) {
        const unsigned long long clave = reg[i].path >> (3 * (nivel - l));
        int lo = 0, hi = n_claves;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (claves[mid] < clave) lo = mid + 1; else hi = mid;
        }
        hallada = (lo < n_claves && claves[lo] == clave);
    }
    if (!hallada) atomicAdd(no_contenidas, 1ULL);
}

__global__ void volumenRegistrosKernel(const RegistroHoja* reg, int n, int lmax,
                                       unsigned long long* vol) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(vol, 1ULL << (3 * (lmax - reg[i].nivel)));
}

bool dagCompararCobertura(const Dag& sub, const Dag& ref, bool verboso,
                          long long* border_difs, long long* inside_no_contenidas,
                          long long* vol_sub_out, long long* vol_ref_out) {
    long long ns = 0, nr = 0;
    RegistroHoja* rs = expandirDag(sub, &ns);
    RegistroHoja* rr = expandirDag(ref, &nr);
    const int lmax = sub.max_nivel > ref.max_nivel ? sub.max_nivel : ref.max_nivel;

    thrust::device_ptr<RegistroHoja> ps(rs), pr(rr);
    EsEstadoReg es_borde{STATE_BORDER}, es_dentro{STATE_INSIDE};
    const int nbs = (int)thrust::count_if(ps, ps + ns, es_borde);
    const int nbr = (int)thrust::count_if(pr, pr + nr, es_borde);
    const int nis = (int)thrust::count_if(ps, ps + ns, es_dentro);
    const int nir = (int)thrust::count_if(pr, pr + nr, es_dentro);

    // 1) Hojas BORDER idénticas.
    unsigned long long difs_borde = 0;
    if (nbs != nbr) {
        difs_borde = (unsigned long long)(nbs > nbr ? nbs - nbr : nbr - nbs);
        if (difs_borde == 0) difs_borde = 1;
    } else if (nbs > 0) {
        RegistroHoja* bs = dalloc<RegistroHoja>(nbs);
        RegistroHoja* br = dalloc<RegistroHoja>(nbr);
        thrust::copy_if(thrust::device, rs, rs + ns, bs, es_borde);
        thrust::copy_if(thrust::device, rr, rr + nr, br, es_borde);
        unsigned long long* d_difs = dalloc<unsigned long long>(1);
        CUDA_CHECK(cudaMemset(d_difs, 0, sizeof(unsigned long long)));
        compararRegistrosKernel<<<bloques(nbs), THREADS>>>(bs, br, nbs, d_difs);
        CUDA_CHECK(cudaGetLastError());
        difs_borde = leerContador(d_difs);
        dfree(d_difs, 1);
        dfree(bs, nbs);
        dfree(br, nbr);
    }

    // 2) Cada hoja INSIDE de sub está contenida en una INSIDE de ref.
    unsigned long long no_cont = 0;
    unsigned long long vol_s = 0, vol_r = 0;
    if (nis > 0 && nir > 0) {
        RegistroHoja* is = dalloc<RegistroHoja>(nis);
        RegistroHoja* ir = dalloc<RegistroHoja>(nir);
        thrust::copy_if(thrust::device, rs, rs + ns, is, es_dentro);
        thrust::copy_if(thrust::device, rr, rr + nr, ir, es_dentro);
        unsigned long long* claves = dalloc<unsigned long long>(nir);
        clavesDeRegistrosKernel<<<bloques(nir), THREADS>>>(ir, nir, claves);
        CUDA_CHECK(cudaGetLastError());   // ya están ordenadas por path
        unsigned long long* d_cnt = dalloc<unsigned long long>(3);
        CUDA_CHECK(cudaMemset(d_cnt, 0, 3 * sizeof(unsigned long long)));
        contenidaPorKernel<<<bloques(nis), THREADS>>>(claves, nir, is, nis, d_cnt);
        volumenRegistrosKernel<<<bloques(nis), THREADS>>>(is, nis, lmax, d_cnt + 1);
        volumenRegistrosKernel<<<bloques(nir), THREADS>>>(ir, nir, lmax, d_cnt + 2);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long h[3];
        CUDA_CHECK(cudaMemcpy(h, d_cnt, sizeof(h), cudaMemcpyDeviceToHost));
        no_cont = h[0]; vol_s = h[1]; vol_r = h[2];
        dfree(d_cnt, 3);
        dfree(claves, nir);
        dfree(is, nis);
        dfree(ir, nir);
    } else if (nis != nir) {
        no_cont = 1;                       // uno tiene INSIDE y el otro no
    }
    dfree(rs, ns > 0 ? ns : 1);
    dfree(rr, nr > 0 ? nr : 1);

    if (border_difs) *border_difs = (long long)difs_borde;
    if (inside_no_contenidas) *inside_no_contenidas = (long long)no_cont;
    if (vol_sub_out) *vol_sub_out = (long long)vol_s;
    if (vol_ref_out) *vol_ref_out = (long long)vol_r;
    const bool ok = (difs_borde == 0 && no_cont == 0 && vol_s == vol_r);
    if (verboso) {
        printf("  Cobertura vs monolítico: BORDER %d vs %d (%llu diferencias); "
               "INSIDE %d vs %d (el DAG por subárboles %s), %llu INSIDE no "
               "contenidas; volumen INSIDE %llu vs %llu (celdas del nivel %d) "
               "-> %s\n", nbs, nbr, difs_borde, nis, nir,
               nis >= nir ? "más subdividido" : "MENOS subdividido", no_cont,
               vol_s, vol_r, lmax, ok ? "OK" : "FALLO");
    }
    return ok;
}
