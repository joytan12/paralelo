#include "octree.cuh"
#include "stats.cuh"
#include <cstdio>
#include <cstdlib>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

// --------------------------------------------------------------------------
// Macro para verificar errores de CUDA de forma concisa.
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
// Representación auxiliar ordenada por clave de camino.
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
// Utilidades de la jerarquía octree.
// --------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long childPathCode(
    unsigned long long parent_code, int child_index) {
    return (parent_code << 3) | (unsigned long long)child_index;
}

__device__ __forceinline__ unsigned long long pathCodeFromGrid(
    int x, int y, int z, int level) {
    // La clave empieza con un 1 implícito para distinguir caminos con ceros
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
// Inicializa la bandera de subdivisión: se refinan los BORDER que todavía
// no alcanzaron el nivel máximo.
// --------------------------------------------------------------------------
__global__ void markInitialSplitsKernel(const Cube* input, int n, int max_level,
                                        int* split) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    split[tid] = (input[tid].state == STATE_BORDER &&
                  input[tid].level < max_level) ? 1 : 0;
}

// --------------------------------------------------------------------------
// Construye las claves de búsqueda de vecinos en device.
// --------------------------------------------------------------------------
__global__ void buildLeafKeysKernel(const Cube* input, int n, LeafKey* keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    keys[tid].path_code = input[tid].path_code;
    keys[tid].input_index = tid;
}

// --------------------------------------------------------------------------
// Marca vecinos más gruesos de cubos seleccionados.
//
// Para un cubo seleccionado de nivel lf, se consultan los niveles menores.
// En cada nivel se generan las pocas celdas cuya AABB puede tocar al cubo y
// se localizan por búsqueda binaria sobre las claves Morton ordenadas.
// Se consideran caras, aristas y vértices, igual que la versión anterior.
// --------------------------------------------------------------------------
__global__ void markCoarserNeighborsKernel(const Cube* input, int n,
                                           int* split,
                                           const LeafKey* sorted_keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || split[tid] == 0) return;

    const Cube fine = input[tid];

    for (int coarse_level = 0; coarse_level < fine.level; ++coarse_level) {
        const int scale = 1 << (fine.level - coarse_level);
        const int level_limit = 1 << coarse_level;

        // En una dimensión, la celda gruesa que toca un intervalo fino puede
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
                        sorted_keys, n, code);
                    if (neighbor < 0 || neighbor == tid) continue;

                    // El vecino es estrictamente más grueso por construcción.
                    atomicExch(&split[neighbor], 1);
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Genera la siguiente lista de hojas usando un exclusive scan de split.
// Si k padres anteriores fueron divididos, la posición de i aumenta 7*k.
// --------------------------------------------------------------------------
__global__ void generateAdaptiveOutputKernel(const Cube* input, int n,
                                             const int* split,
                                             const int* split_prefix,
                                             Cube* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

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
// Refinamiento adaptativo y balance 2:1 completamente en GPU.
//
// No se copia ningún Cube al host. Los únicos valores que regresan son
// contadores enteros usados por el host para configurar la siguiente etapa.
// --------------------------------------------------------------------------
void launchAdaptiveRefinement(const Cube* d_input, int n_input, int max_level,
                              Cube** d_output, int* n_output,
                              int* n_refined_parents) {
    *d_output = nullptr;
    *n_output = n_input;
    *n_refined_parents = 0;

    if (n_input <= 0) return;

    const int threads = 256;
    const int blocks = (n_input + threads - 1) / threads;

    int* d_split = nullptr;
    int* d_split_prefix = nullptr;
    LeafKey* d_keys = nullptr;

    CUDA_CHECK(cudaMalloc(&d_split, n_input * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_split_prefix, n_input * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys, n_input * sizeof(LeafKey)));

    markInitialSplitsKernel<<<blocks, threads>>>(
        d_input, n_input, max_level, d_split);
    CUDA_CHECK(cudaGetLastError());

    buildLeafKeysKernel<<<blocks, threads>>>(d_input, n_input, d_keys);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<LeafKey> keys_begin(d_keys);
    thrust::device_ptr<LeafKey> keys_end(d_keys + n_input);
    thrust::sort(keys_begin, keys_end, LeafKeyLess());

    // Cierre iterativo del balance. Se ejecuta un número acotado de rondas
    // suficiente para propagar una marca desde cualquier nivel hasta la
    // raíz. No hay flags ni cubos que vuelvan al host.
    for (int iteration = 0; iteration < max_level; ++iteration) {
        markCoarserNeighborsKernel<<<blocks, threads>>>(
            d_input, n_input, d_split, d_keys);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    thrust::device_ptr<int> split_begin(d_split);
    thrust::device_ptr<int> split_end(d_split + n_input);
    const int n_split = (int)thrust::count(split_begin, split_end, 1);
    *n_refined_parents = n_split;

    if (n_split == 0) {
        CUDA_CHECK(cudaFree(d_split));
        CUDA_CHECK(cudaFree(d_split_prefix));
        CUDA_CHECK(cudaFree(d_keys));
        return;
    }

    *n_output = n_input + 7 * n_split;
    CUDA_CHECK(cudaMalloc(d_output, (*n_output) * sizeof(Cube)));

    thrust::device_ptr<int> prefix_begin(d_split_prefix);
    thrust::exclusive_scan(split_begin, split_end, prefix_begin);

    generateAdaptiveOutputKernel<<<blocks, threads>>>(
        d_input, n_input, d_split, d_split_prefix, *d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Punto de mayor ocupacion de la etapa: coexisten la lista de
    // entrada, la de salida y los tres arreglos auxiliares.
    gpuSampleVram();

    CUDA_CHECK(cudaFree(d_split));
    CUDA_CHECK(cudaFree(d_split_prefix));
    CUDA_CHECK(cudaFree(d_keys));
}

// --------------------------------------------------------------------------
// Función pública de refinamiento uniforme.
// --------------------------------------------------------------------------
void launchRefinement(const Cube* d_input, int n_input,
                      Cube** d_output, int* n_output) {
    *n_output = n_input * 8;
    CUDA_CHECK(cudaMalloc(d_output, (*n_output) * sizeof(Cube)));

    const int threads = 256;
    const int blocks = (n_input + threads - 1) / threads;
    refineKernel<<<blocks, threads>>>(d_input, n_input, *d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Entrada y salida coexisten: pico de la etapa uniforme.
    gpuSampleVram();
}

// --------------------------------------------------------------------------
// Poda GPU: conserva cubos INSIDE y BORDER.
// --------------------------------------------------------------------------
struct IsNotOutside {
    __host__ __device__ bool operator()(const Cube& c) const {
        return c.state != STATE_OUTSIDE;
    }
};

void launchPrune(const Cube* d_input, int n_input,
                 Cube** d_output, int* n_output) {
    CUDA_CHECK(cudaMalloc(d_output, n_input * sizeof(Cube)));

    // La poda reserva el peor caso, asi que aqui conviven dos arreglos
    // completos. Suele ser el punto mas alto de toda la ejecucion.
    gpuSampleVram();

    thrust::device_ptr<const Cube> in_begin(d_input);
    thrust::device_ptr<const Cube> in_end(d_input + n_input);
    thrust::device_ptr<Cube> out_begin(*d_output);
    thrust::device_ptr<Cube> out_end =
        thrust::copy_if(in_begin, in_end, out_begin, IsNotOutside());

    *n_output = (int)(out_end - out_begin);
}

// --------------------------------------------------------------------------
// Conteo por estado en device.
//
// Se usa para llenar la tabla de tiempo y espacio por nivel sin traer los
// cubos a la RAM: solo cruza el bus el entero resultante.
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
// Memoria auxiliar del refinamiento adaptativo, por hoja de entrada.
// --------------------------------------------------------------------------
size_t adaptiveScratchBytesPerLeaf() {
    // d_split (int) + d_split_prefix (int) + d_keys (LeafKey)
    return 2 * sizeof(int) + sizeof(LeafKey);
}
