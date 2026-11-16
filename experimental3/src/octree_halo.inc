// ==========================================================================
// Hito 2b: refinamiento adaptativo con halo, cáscaras y verificación 2:1.
//
// Este archivo se incluye al final de octree.cu y comparte sus utilidades
// (LeafKey, findLeafByPath, pathCodeFromGrid, makeChildDevice). El mecanismo
// está portado de experimental/src/octree.cu; las funciones del refinamiento
// base (launchAdaptiveRefinement, launchPrune) no se tocan para que el camino
// monolítico siga siendo un oráculo independiente.
// ==========================================================================

#include <thrust/fill.h>
#include <thrust/remove.h>

// --------------------------------------------------------------------------
// Acceso unificado a la lista de trabajo [propias | ghosts].
// --------------------------------------------------------------------------
__device__ __forceinline__ const Cube& fetchCubeHalo(const Cube* own, int n_own,
                                                     const Cube* ghost, int i) {
    return (i < n_own) ? own[i] : ghost[i - n_own];
}

// Bandera inicial: para las hojas propias, BORDER que no alcanzó el nivel
// máximo; para los ghosts, la bandera que su dueño calculó en este mismo
// paso (puede venir del balance interno de su subárbol y no se puede
// reconstruir desde state y level).
__global__ void markInitialSplitsHaloKernel(const Cube* own, int n_own,
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

__global__ void buildLeafKeysHaloKernel(const Cube* own, int n_own,
                                        const Cube* ghost, int n_work,
                                        LeafKey* keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_work) return;
    keys[tid].path_code = fetchCubeHalo(own, n_own, ghost, tid).path_code;
    keys[tid].input_index = tid;
}

// Marca vecinos más gruesos de los cubos seleccionados; como la lista incluye
// los ghosts, un vecino de otro subárbol se encuentra igual que uno propio.
// d_changed se levanta si la ronda cambió alguna bandera.
__global__ void markCoarserNeighborsHaloKernel(const Cube* own, int n_own,
                                               const Cube* ghost, int n_work,
                                               int* split,
                                               const LeafKey* sorted_keys,
                                               int* d_changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_work || split[tid] == 0) return;

    const Cube fine = fetchCubeHalo(own, n_own, ghost, tid);

    for (int coarse_level = 0; coarse_level < fine.level; ++coarse_level) {
        const int scale = 1 << (fine.level - coarse_level);
        const int level_limit = 1 << coarse_level;

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
                    const int neighbor = findLeafByPath(sorted_keys, n_work, code);
                    if (neighbor < 0 || neighbor == tid) continue;

                    if (atomicExch(&split[neighbor], 1) == 0)
                        *d_changed = 1;
                }
            }
        }
    }
}

// Solo recorre la porción propia: los ghosts nunca se materializan.
__global__ void generateAdaptiveOutputHaloKernel(const Cube* input, int n_own,
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

void launchAdaptiveRefinementHalo(const Cube* d_own, int n_own,
                                  const Cube* d_ghost, int n_ghost,
                                  const int* d_ghost_split, int max_level,
                                  Cube** d_output, int* n_output,
                                  int* n_refined_parents, int** d_split_own) {
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

    markInitialSplitsHaloKernel<<<blocks_work, threads>>>(
        d_own, n_own, d_ghost_split, n_work, max_level, d_split);
    CUDA_CHECK(cudaGetLastError());
    buildLeafKeysHaloKernel<<<blocks_work, threads>>>(
        d_own, n_own, d_ghost, n_work, d_keys);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<LeafKey> keys_begin(d_keys);
    thrust::sort(keys_begin, keys_begin + n_work, LeafKeyLess());

    // Cierre iterativo del balance, cortado al alcanzar el punto fijo.
    for (int iteration = 0; iteration < max_level; ++iteration) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        markCoarserNeighborsHaloKernel<<<blocks_work, threads>>>(
            d_own, n_own, d_ghost, n_work, d_split, d_keys, d_changed);
        CUDA_CHECK(cudaGetLastError());
        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (!h_changed) break;
    }

    // Solo cuentan los padres propios.
    thrust::device_ptr<int> split_begin(d_split);
    const int n_split = (int)thrust::count(split_begin, split_begin + n_own, 1);
    *n_refined_parents = n_split;

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_changed));

    if (n_split == 0) {
        if (d_split_own) *d_split_own = d_split;
        else             CUDA_CHECK(cudaFree(d_split));
        CUDA_CHECK(cudaFree(d_split_prefix));
        return;
    }

    *n_output = n_own + 7 * n_split;
    CUDA_CHECK(cudaMalloc(d_output, (size_t)(*n_output) * sizeof(Cube)));

    thrust::device_ptr<int> prefix_begin(d_split_prefix);
    thrust::exclusive_scan(split_begin, split_begin + n_own, prefix_begin);

    generateAdaptiveOutputHaloKernel<<<blocks_own, threads>>>(
        d_own, n_own, d_split, d_split_prefix, *d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    gpuSampleVram();

    if (d_split_own) *d_split_own = d_split;
    else             CUDA_CHECK(cudaFree(d_split));
    CUDA_CHECK(cudaFree(d_split_prefix));
}

// --------------------------------------------------------------------------
// Poda in-place (remove_if estable): no reserva un segundo arreglo.
// --------------------------------------------------------------------------
struct IsOutsideHalo {
    __host__ __device__ bool operator()(const Cube& c) const {
        return c.state == STATE_OUTSIDE;
    }
};

void launchPruneInPlace(Cube* d_inout, int n_input, int* n_output) {
    *n_output = 0;
    if (n_input <= 0) return;
    thrust::device_ptr<Cube> begin(d_inout);
    thrust::device_ptr<Cube> new_end =
        thrust::remove_if(begin, begin + n_input, IsOutsideHalo());
    *n_output = (int)(new_end - begin);
    gpuSampleVram();
}

// --------------------------------------------------------------------------
// Cáscara de un subárbol: hojas que tocan su frontera exterior.
// --------------------------------------------------------------------------
struct IsOnShell {
    int split_level;
    __host__ __device__ bool operator()(const Cube& c) const {
        const int shift = c.level - split_level;
        if (shift <= 0) return true;             // la raíz del subárbol
        const int mask = (1 << shift) - 1;
        const int lx = c.grid_index.x & mask;
        const int ly = c.grid_index.y & mask;
        const int lz = c.grid_index.z & mask;
        return lx == 0 || lx == mask || ly == 0 || ly == mask ||
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

    // Las banderas se filtran con el mismo stencil y predicado: la selección
    // y el orden coinciden celda a celda.
    thrust::device_ptr<int> out_split(*d_shell_split);
    if (d_split) {
        thrust::device_ptr<const int> split_begin(d_split);
        thrust::copy_if(split_begin, split_begin + n_input, begin, out_split,
                        predicate);
    } else {
        thrust::fill(out_split, out_split + n_keep, 0);
    }
}

// --------------------------------------------------------------------------
// Verificación de la regla 2:1 sobre un conjunto de hojas (las cáscaras):
// para cada hoja, sus 26 vecinos del mismo nivel y, subiendo por ancestros,
// si el que los cubre es una hoja 2 o más niveles más gruesa.
// --------------------------------------------------------------------------
__global__ void verifyBalanceHaloKernel(const Cube* leaves, int n,
                                        const LeafKey* keys, int n_keys,
                                        int split_level,
                                        unsigned long long* violations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const Cube fine = leaves[tid];
    const int la = fine.level;
    if (la - 2 < split_level) return;

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

    buildLeafKeysKernel<<<blocks, threads>>>(d_leaves, n_leaves, d_keys);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<LeafKey> kb(d_keys);
    thrust::sort(kb, kb + n_leaves, LeafKeyLess());

    verifyBalanceHaloKernel<<<blocks, threads>>>(
        d_leaves, n_leaves, d_keys, n_leaves, split_level, d_violations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h = 0ULL;
    CUDA_CHECK(cudaMemcpy(&h, d_violations, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_violations));
    return h;
}
