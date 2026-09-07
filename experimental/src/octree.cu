#include "octree.cuh"
#include "stats.cuh"
#include <cstdio>
#include <cstdlib>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/fill.h>

// --------------------------------------------------------------------------
// Macro para verificar errores de CUDA de forma concisa.
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
// Representacion auxiliar ordenada por clave de camino.
// --------------------------------------------------------------------------
struct LeafKey {
    unsigned long long path_code;
    int input_index;
};

struct LeafKeyLess {
    __host__ __device__ bool operator()(const LeafKey& a,
                                        const LeafKey& b) const {
        return a.path_code < b.path_code;
    }
};

// --------------------------------------------------------------------------
// Acceso unificado a la lista de trabajo.
//
// La lista logica es la concatenacion [propias | ghosts]. Se resuelve con un
// indice en vez de copiar: los ghosts son de otro subarbol y solo se leen.
// --------------------------------------------------------------------------
__device__ __forceinline__ const Cube& fetchCube(const Cube* own, int n_own,
                                                 const Cube* ghost, int i) {
    return (i < n_own) ? own[i] : ghost[i - n_own];
}

// --------------------------------------------------------------------------
// Utilidades de la jerarquia octree.
// --------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long childPathCode(
    unsigned long long parent_code, int child_index) {
    return (parent_code << 3) | (unsigned long long)child_index;
}

__device__ __forceinline__ unsigned long long pathCodeFromGrid(
    int x, int y, int z, int level) {
    // La clave empieza con un 1 implicito para distinguir caminos con ceros
    // iniciales: root=1, root-child-0=8, root-child-0-child-0=64, etc.
    unsigned long long code = 1ULL;
    for (int bit = level - 1; bit >= 0; --bit) {
        int octant = ((x >> bit) & 1) |
                     (((y >> bit) & 1) << 1) |
                     (((z >> bit) & 1) << 2);
        code = (code << 3) | (unsigned long long)octant;
    }
    return code;
}

__device__ __forceinline__ int findLeafByPath(
    const LeafKey* keys, int n_keys, unsigned long long path_code) {
    int first = 0;
    int last = n_keys;
    while (first < last) {
        int middle = first + (last - first) / 2;
        unsigned long long candidate = keys[middle].path_code;
        if (candidate < path_code) {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    if (first < n_keys && keys[first].path_code == path_code)
        return keys[first].input_index;
    return -1;
}

__device__ __forceinline__ Cube makeChildDevice(const Cube& parent,
                                                int child_index) {
    const float child_half = parent.half_size * 0.5f;
    Cube child;
    child.center.x = parent.center.x +
                     ((child_index & 1) ? child_half : -child_half);
    child.center.y = parent.center.y +
                     ((child_index & 2) ? child_half : -child_half);
    child.center.z = parent.center.z +
                     ((child_index & 4) ? child_half : -child_half);
    child.half_size = child_half;
    child.level = parent.level + 1;
    child.state = STATE_UNCLASSIFIED;
    child.grid_index = make_int3(
        parent.grid_index.x * 2 + ((child_index >> 0) & 1),
        parent.grid_index.y * 2 + ((child_index >> 1) & 1),
        parent.grid_index.z * 2 + ((child_index >> 2) & 1));
    child.path_code = childPathCode(parent.path_code, child_index);
    return child;
}

// --------------------------------------------------------------------------
// Refinamiento uniforme: cada hilo procesa un cubo y genera 8 hijos.
// --------------------------------------------------------------------------
__global__ void refineKernel(const Cube* input, int n, Cube* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const Cube parent = input[tid];
    const int base = tid * 8;
    for (int child = 0; child < 8; ++child)
        output[base + child] = makeChildDevice(parent, child);
}

// --------------------------------------------------------------------------
// Bandera inicial de subdivision: BORDER que no alcanzo el nivel maximo.
//
// Para las hojas propias es el criterio geometrico del proyecto base. Para
// los ghosts se usa la bandera que su dueno calculo en este mismo paso: un
// ghost puede estar marcado por el balance interno de su subarbol, y esa
// marca no se puede reconstruir desde state y level. Sin ella se corta la
// cadena "celda fina interior de A -> hoja de frontera de A -> hoja gruesa
// de B" y el balance 2:1 queda roto justo en la frontera.
// --------------------------------------------------------------------------
__global__ void markInitialSplitsKernel(const Cube* own, int n_own,
                                        const Cube* ghost,
                                        const int* ghost_split, int n_work,
                                        int max_level, int* split) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_work) return;

    if (tid >= n_own) {
        split[tid] = ghost_split ? ghost_split[tid - n_own] : 0;
        return;
    }
    const Cube& c = own[tid];
    split[tid] = (c.state == STATE_BORDER && c.level < max_level) ? 1 : 0;
}

// --------------------------------------------------------------------------
// Construye las claves de busqueda de vecinos en device.
// --------------------------------------------------------------------------
__global__ void buildLeafKeysKernel(const Cube* own, int n_own,
                                    const Cube* ghost, int n_work,
                                    LeafKey* keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_work) return;
    keys[tid].path_code = fetchCube(own, n_own, ghost, tid).path_code;
    keys[tid].input_index = tid;
}

// --------------------------------------------------------------------------
// Marca vecinos mas gruesos de cubos seleccionados.
//
// Para un cubo seleccionado de nivel lf se consultan los niveles menores. En
// cada nivel se generan las pocas celdas cuya AABB puede tocarlo y se
// localizan por busqueda binaria sobre las claves ordenadas. Como la lista
// incluye los ghosts, un vecino que vive en otro subarbol se encuentra igual
// que uno propio y la regla 2:1 se cumple a traves de la frontera.
//
// d_changed se levanta cuando esta ronda cambio al menos una bandera, para
// poder cortar el cierre iterativo apenas alcanza el punto fijo.
// --------------------------------------------------------------------------
__global__ void markCoarserNeighborsKernel(const Cube* own, int n_own,
                                           const Cube* ghost, int n_work,
                                           int* split,
                                           const LeafKey* sorted_keys,
                                           int* d_changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_work || split[tid] == 0) return;

    const Cube fine = fetchCube(own, n_own, ghost, tid);

    for (int coarse_level = 0; coarse_level < fine.level; ++coarse_level) {
        const int scale = 1 << (fine.level - coarse_level);
        const int level_limit = 1 << coarse_level;

        // En una dimension, la celda gruesa que toca un intervalo fino puede
        // estar entre floor(start/scale)-1 y floor(end/scale).
        const int min_x = (fine.grid_index.x + scale - 1) / scale - 1;
        const int max_x = (fine.grid_index.x + 1) / scale;
        const int min_y = (fine.grid_index.y + scale - 1) / scale - 1;
        const int max_y = (fine.grid_index.y + 1) / scale;
        const int min_z = (fine.grid_index.z + scale - 1) / scale - 1;
        const int max_z = (fine.grid_index.z + 1) / scale;

        for (int x = min_x; x <= max_x; ++x) {
            if (x < 0 || x >= level_limit) continue;
            for (int y = min_y; y <= max_y; ++y) {
                if (y < 0 || y >= level_limit) continue;
                for (int z = min_z; z <= max_z; ++z) {
                    if (z < 0 || z >= level_limit) continue;

                    const unsigned long long code =
                        pathCodeFromGrid(x, y, z, coarse_level);
                    const int neighbor = findLeafByPath(
                        sorted_keys, n_work, code);
                    if (neighbor < 0 || neighbor == tid) continue;

                    // El vecino es estrictamente mas grueso por construccion.
                    if (atomicExch(&split[neighbor], 1) == 0)
                        *d_changed = 1;
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Genera la siguiente lista de hojas usando un exclusive scan de split.
// Si k padres anteriores fueron divididos, la posicion de i aumenta 7*k.
//
// Solo recorre la porcion propia: los ghosts nunca se materializan.
// --------------------------------------------------------------------------
__global__ void generateAdaptiveOutputKernel(const Cube* input, int n_own,
                                             const int* split,
                                             const int* split_prefix,
                                             Cube* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_own) return;

    const int output_base = tid + 7 * split_prefix[tid];
    if (split[tid] == 0) {
        output[output_base] = input[tid];
        return;
    }

    const Cube parent = input[tid];
    for (int child = 0; child < 8; ++child)
        output[output_base + child] = makeChildDevice(parent, child);
}

// --------------------------------------------------------------------------
// Refinamiento adaptativo con balance 2:1 y halo, completamente en GPU.
// --------------------------------------------------------------------------
void launchAdaptiveRefinement(const Cube* d_own, int n_own,
                              const Cube* d_ghost, int n_ghost,
                              const int* d_ghost_split,
                              int max_level,
                              Cube** d_output, int* n_output,
                              int* n_refined_parents,
                              int** d_split_own) {
    *d_output = nullptr;
    *n_output = n_own;
    *n_refined_parents = 0;
    if (d_split_own) *d_split_own = nullptr;

    if (n_own <= 0) return;
    if (n_ghost < 0) n_ghost = 0;

    const int n_work = n_own + n_ghost;
    const int threads = 256;
    const int blocks_work = (n_work + threads - 1) / threads;
    const int blocks_own  = (n_own  + threads - 1) / threads;

    int* d_split = nullptr;
    int* d_split_prefix = nullptr;
    int* d_changed = nullptr;
    LeafKey* d_keys = nullptr;

    CUDA_CHECK(cudaMalloc(&d_split, (size_t)n_work * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_split_prefix, (size_t)n_own * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys, (size_t)n_work * sizeof(LeafKey)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    markInitialSplitsKernel<<<blocks_work, threads>>>(
        d_own, n_own, d_ghost, d_ghost_split, n_work, max_level, d_split);
    CUDA_CHECK(cudaGetLastError());

    buildLeafKeysKernel<<<blocks_work, threads>>>(
        d_own, n_own, d_ghost, n_work, d_keys);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<LeafKey> keys_begin(d_keys);
    thrust::device_ptr<LeafKey> keys_end(d_keys + n_work);
    thrust::sort(keys_begin, keys_end, LeafKeyLess());

    // Cierre iterativo del balance. El limite superior sigue siendo max_level
    // rondas (basta para propagar una marca desde cualquier nivel hasta la
    // raiz), pero se corta apenas una ronda no cambia ninguna bandera: es el
    // mismo punto fijo, alcanzado antes.
    for (int iteration = 0; iteration < max_level; ++iteration) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        markCoarserNeighborsKernel<<<blocks_work, threads>>>(
            d_own, n_own, d_ghost, n_work, d_split, d_keys, d_changed);
        CUDA_CHECK(cudaGetLastError());

        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (!h_changed) break;
    }

    // Solo cuentan los padres propios: un ghost marcado lo subdivide su
    // dueno cuando le toque, no nosotros.
    thrust::device_ptr<int> split_begin(d_split);
    const int n_split = (int)thrust::count(split_begin, split_begin + n_own, 1);
    *n_refined_parents = n_split;

    // d_keys y d_changed ya cumplieron su proposito (busqueda de vecinos y
    // corte del cierre iterativo). Se liberan aqui, antes de reservar la
    // salida, en vez de al final de la funcion: en el peor caso (subarbol
    // grande, poco margen) esta espera de mas es justo lo que decide si la
    // reserva de *d_output cabe en VRAM fisica o no.
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_changed));

    if (n_split == 0) {
        if (d_split_own) *d_split_own = d_split;
        else             CUDA_CHECK(cudaFree(d_split));
        CUDA_CHECK(cudaFree(d_split_prefix));
        return;
    }

    *n_output = n_own + 7 * n_split;
    gpuRequireBudget((size_t)(*n_output) * sizeof(Cube),
                     "refinamiento adaptativo (salida)");
    CUDA_CHECK(cudaMalloc(d_output, (size_t)(*n_output) * sizeof(Cube)));

    thrust::device_ptr<int> prefix_begin(d_split_prefix);
    thrust::exclusive_scan(split_begin, split_begin + n_own, prefix_begin);

    generateAdaptiveOutputKernel<<<blocks_own, threads>>>(
        d_own, n_own, d_split, d_split_prefix, *d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Punto de mayor ocupacion de la etapa: coexisten la lista de entrada, la
    // de salida, el halo y los arreglos auxiliares.
    gpuSampleVram();

    if (d_split_own) *d_split_own = d_split;
    else             CUDA_CHECK(cudaFree(d_split));
    CUDA_CHECK(cudaFree(d_split_prefix));
}

// --------------------------------------------------------------------------
// Funcion publica de refinamiento uniforme.
// --------------------------------------------------------------------------
void launchRefinement(const Cube* d_input, int n_input,
                      Cube** d_output, int* n_output) {
    *n_output = n_input * 8;
    gpuRequireBudget((size_t)(*n_output) * sizeof(Cube),
                     "refinamiento uniforme (salida)");
    CUDA_CHECK(cudaMalloc(d_output, (size_t)(*n_output) * sizeof(Cube)));

    const int threads = 256;
    const int blocks = (n_input + threads - 1) / threads;
    refineKernel<<<blocks, threads>>>(d_input, n_input, *d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    gpuSampleVram();
}

// --------------------------------------------------------------------------
// Poda GPU IN-PLACE: conserva cubos INSIDE y BORDER, elimina FUERA.
//
// El proyecto base (y una version anterior de este) reservaban un buffer de
// salida aparte -- el peor caso en el base, el tamano exacto aqui -- lo que
// de todas formas obligaba a que entrada y salida coexistieran en VRAM
// durante la poda. thrust::remove_if compacta dentro del mismo arreglo
// (orden estable, igual que std::remove_if): no hay una segunda copia, asi
// que el pico de la poda deja de ser un multiplo de n_input y pasa a ser
// exactamente n_input -- lo que ya estaba reservado por el refinamiento.
// --------------------------------------------------------------------------
struct IsOutside {
    __host__ __device__ bool operator()(const Cube& c) const {
        return c.state == STATE_OUTSIDE;
    }
};

void launchPrune(Cube* d_inout, int n_input, int* n_output) {
    *n_output = 0;
    if (n_input <= 0) return;

    thrust::device_ptr<Cube> begin(d_inout);
    thrust::device_ptr<Cube> end(d_inout + n_input);

    thrust::device_ptr<Cube> new_end =
        thrust::remove_if(begin, end, IsOutside());
    *n_output = (int)(new_end - begin);

    // Sin reserva nueva: el pico de la poda es el mismo arreglo de entrada.
    gpuSampleVram();
}

// --------------------------------------------------------------------------
// Conteo por estado en device.
// --------------------------------------------------------------------------
struct HasState {
    int target;
    __host__ __device__ bool operator()(const Cube& c) const {
        return c.state == target;
    }
};

int launchCountByState(const Cube* d_input, int n_input, int target_state) {
    if (n_input <= 0) return 0;

    thrust::device_ptr<const Cube> begin(d_input);
    thrust::device_ptr<const Cube> end(d_input + n_input);
    HasState predicate;
    predicate.target = target_state;
    return (int)thrust::count_if(begin, end, predicate);
}

// --------------------------------------------------------------------------
// Cascara de un subarbol.
//
// Una hoja de nivel L cuyo ancestro de nivel D define el subarbol ocupa,
// dentro de ese subarbol, coordenadas locales de 0 a 2^(L-D)-1. Toca la
// frontera exterior si alguna coordenada local esta en un extremo.
// --------------------------------------------------------------------------
struct IsOnShell {
    int split_level;
    __host__ __device__ bool operator()(const Cube& c) const {
        const int shift = c.level - split_level;
        if (shift <= 0) return true;             // la raiz del subarbol
        const int mask = (1 << shift) - 1;
        const int lx = c.grid_index.x & mask;
        const int ly = c.grid_index.y & mask;
        const int lz = c.grid_index.z & mask;
        return lx == 0 || lx == mask ||
               ly == 0 || ly == mask ||
               lz == 0 || lz == mask;
    }
};

void launchExtractShell(const Cube* d_input, const int* d_split, int n_input,
                        int split_level,
                        Cube** d_shell, int** d_shell_split, int* n_shell) {
    *d_shell       = nullptr;
    *d_shell_split = nullptr;
    *n_shell       = 0;
    if (n_input <= 0) return;

    IsOnShell predicate;
    predicate.split_level = split_level;

    thrust::device_ptr<const Cube> begin(d_input);
    thrust::device_ptr<const Cube> end(d_input + n_input);

    const int n_keep = (int)thrust::count_if(begin, end, predicate);
    *n_shell = n_keep;
    if (n_keep == 0) return;

    CUDA_CHECK(cudaMalloc(d_shell, (size_t)n_keep * sizeof(Cube)));
    CUDA_CHECK(cudaMalloc(d_shell_split, (size_t)n_keep * sizeof(int)));

    thrust::device_ptr<Cube> out(*d_shell);
    thrust::copy_if(begin, end, out, predicate);

    // Las banderas se filtran usando el mismo arreglo de cubos como stencil y
    // el mismo predicado: la seleccion y el orden coinciden celda a celda.
    thrust::device_ptr<int> out_split(*d_shell_split);
    if (d_split) {
        thrust::device_ptr<const int> split_begin(d_split);
        thrust::copy_if(split_begin, split_begin + n_input, begin,
                        out_split, predicate);
    } else {
        thrust::fill(out_split, out_split + n_keep, 0);
    }
}

// --------------------------------------------------------------------------
// Verificacion de la regla 2:1.
//
// Para cada hoja fina se recorren sus 26 vecinos de su mismo nivel y se
// busca, subiendo por los ancestros, si el que los cubre es una hoja dos o
// mas niveles mas gruesa. Cada hallazgo es una violacion.
// --------------------------------------------------------------------------
__global__ void verifyBalanceKernel(const Cube* leaves, int n,
                                    const LeafKey* keys, int n_keys,
                                    int split_level,
                                    unsigned long long* violations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const Cube fine = leaves[tid];
    const int la = fine.level;
    if (la - 2 < split_level) return;   // no cabe un vecino 2 niveles mas grueso

    const int limit = 1 << la;

    for (int dx = -1; dx <= 1; ++dx) {
        const int nx = fine.grid_index.x + dx;
        if (nx < 0 || nx >= limit) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            const int ny = fine.grid_index.y + dy;
            if (ny < 0 || ny >= limit) continue;
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                const int nz = fine.grid_index.z + dz;
                if (nz < 0 || nz >= limit) continue;

                for (int cl = la - 2; cl >= split_level; --cl) {
                    const int sh = la - cl;
                    const unsigned long long code =
                        pathCodeFromGrid(nx >> sh, ny >> sh, nz >> sh, cl);
                    if (findLeafByPath(keys, n_keys, code) >= 0) {
                        atomicAdd(violations, 1ULL);
                        break;
                    }
                }
            }
        }
    }
}

__global__ void buildPlainKeysKernel(const Cube* leaves, int n, LeafKey* keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    keys[tid].path_code = leaves[tid].path_code;
    keys[tid].input_index = tid;
}

unsigned long long launchVerifyBalance(const Cube* d_leaves, int n_leaves,
                                       int split_level) {
    if (n_leaves <= 0) return 0ULL;

    const int threads = 256;
    const int blocks = (n_leaves + threads - 1) / threads;

    LeafKey* d_keys = nullptr;
    unsigned long long* d_violations = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, (size_t)n_leaves * sizeof(LeafKey)));
    CUDA_CHECK(cudaMalloc(&d_violations, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_violations, 0, sizeof(unsigned long long)));

    buildPlainKeysKernel<<<blocks, threads>>>(d_leaves, n_leaves, d_keys);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<LeafKey> kb(d_keys);
    thrust::sort(kb, kb + n_leaves, LeafKeyLess());

    verifyBalanceKernel<<<blocks, threads>>>(
        d_leaves, n_leaves, d_keys, n_leaves, split_level, d_violations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h_violations = 0ULL;
    CUDA_CHECK(cudaMemcpy(&h_violations, d_violations,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_violations));
    return h_violations;
}

// --------------------------------------------------------------------------
// Memoria auxiliar del refinamiento adaptativo, por hoja de entrada.
// --------------------------------------------------------------------------
size_t adaptiveScratchBytesPerLeaf() {
    // d_split (int) + d_split_prefix (int). d_keys (LeafKey) ya no cuenta:
    // se libera antes de reservar la salida, asi que no coexiste con ella
    // en el momento de mayor ocupacion.
    return 2 * sizeof(int);
}
