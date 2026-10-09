#include "merge.cuh"

#include <climits>
#include <cstdio>
#include <cstdlib>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error en %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Clave ordenable para encontrar en O(log n) una hoja por su camino Morton.
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

__device__ __forceinline__ unsigned long long pathCodeFromGrid(
    int x, int y, int z, int level) {
    unsigned long long code = 1ULL;
    for (int bit = level - 1; bit >= 0; --bit) {
        const int octant = ((x >> bit) & 1) |
                           (((y >> bit) & 1) << 1) |
                           (((z >> bit) & 1) << 2);
        code = (code << 3) | static_cast<unsigned long long>(octant);
    }
    return code;
}

__device__ __forceinline__ int findLeafByPath(const LeafKey* keys, int n,
                                               unsigned long long path_code) {
    int first = 0;
    int last = n;
    while (first < last) {
        const int middle = first + (last - first) / 2;
        if (keys[middle].path_code < path_code) first = middle + 1;
        else last = middle;
    }
    return (first < n && keys[first].path_code == path_code)
        ? keys[first].input_index : -1;
}

// Busca una hoja en la celda vecina del mismo nivel. La búsqueda exacta por
// path_code hace que una hoja más fina o más gruesa nunca se una a esta.
__device__ __forceinline__ int sameLevelNeighbor(const Cube& cube, int dx,
                                                 int dy, int dz,
                                                 const LeafKey* keys, int n) {
    const int limit = 1 << cube.level;
    const int x = cube.grid_index.x + dx;
    const int y = cube.grid_index.y + dy;
    const int z = cube.grid_index.z + dz;
    if (x < 0 || x >= limit || y < 0 || y >= limit || z < 0 || z >= limit)
        return -1;
    return findLeafByPath(keys, n, pathCodeFromGrid(x, y, z, cube.level));
}

__global__ void buildMergeLeafKeysKernel(const Cube* cubes, int n, LeafKey* keys) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    keys[tid].path_code = cubes[tid].path_code;
    keys[tid].input_index = tid;
}

__global__ void initializeParentsKernel(int* parent, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) parent[tid] = tid;
}

__device__ __forceinline__ int findRoot(const int* parent, int value) {
    while (parent[value] != value) value = parent[value];
    return value;
}

__device__ __forceinline__ void unite(int* parent, int a, int b) {
    while (true) {
        int root_a = findRoot(parent, a);
        int root_b = findRoot(parent, b);
        if (root_a == root_b) return;

        const int low = root_a < root_b ? root_a : root_b;
        const int high = root_a < root_b ? root_b : root_a;
        // Solo se modifica una raíz. Como siempre apunta al índice menor, el
        // bosque permanece acíclico incluso cuando muchos hilos unen a la vez.
        if (atomicCAS(&parent[high], high, low) == high) return;
    }
}

// Solo mira las direcciones positivas: cada arista de conectividad se une una
// vez. La condición de nivel ya la impone sameLevelNeighbor().
__global__ void unionSameLevelNeighborsKernel(const Cube* cubes, int n,
                                               const LeafKey* keys,
                                               int* parent) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const Cube cube = cubes[tid];
    const int nx = sameLevelNeighbor(cube, 1, 0, 0, keys, n);
    const int ny = sameLevelNeighbor(cube, 0, 1, 0, keys, n);
    const int nz = sameLevelNeighbor(cube, 0, 0, 1, keys, n);
    if (nx >= 0) unite(parent, tid, nx);
    if (ny >= 0) unite(parent, tid, ny);
    if (nz >= 0) unite(parent, tid, nz);
}

__global__ void compressParentsKernel(int* parent, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    parent[tid] = parent[parent[tid]];
}

__device__ __forceinline__ unsigned char faceMask(const Cube& cube, int n,
                                                   const LeafKey* keys) {
    unsigned char mask = 0;
    if (sameLevelNeighbor(cube, -1, 0, 0, keys, n) < 0) mask |= 1u << FACE_X_NEG;
    if (sameLevelNeighbor(cube,  1, 0, 0, keys, n) < 0) mask |= 1u << FACE_X_POS;
    if (sameLevelNeighbor(cube, 0, -1, 0, keys, n) < 0) mask |= 1u << FACE_Y_NEG;
    if (sameLevelNeighbor(cube, 0,  1, 0, keys, n) < 0) mask |= 1u << FACE_Y_POS;
    if (sameLevelNeighbor(cube, 0, 0, -1, keys, n) < 0) mask |= 1u << FACE_Z_NEG;
    if (sameLevelNeighbor(cube, 0, 0,  1, keys, n) < 0) mask |= 1u << FACE_Z_POS;
    return mask;
}

__global__ void markExteriorFacesKernel(const Cube* cubes, int n,
                                         const LeafKey* keys,
                                         unsigned char* masks, int* counts) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const unsigned char mask = faceMask(cubes[tid], n, keys);
    masks[tid] = mask;
    counts[tid] = __popc(static_cast<unsigned int>(mask));
}

__device__ __forceinline__ void setFacePoints(MergedFace& face, const Cube& c,
                                               int direction) {
    const float x0 = c.center.x - c.half_size;
    const float x1 = c.center.x + c.half_size;
    const float y0 = c.center.y - c.half_size;
    const float y1 = c.center.y + c.half_size;
    const float z0 = c.center.z - c.half_size;
    const float z1 = c.center.z + c.half_size;

    // El orden de cada cuadrilátero mira hacia el exterior.
    switch (direction) {
        case FACE_X_NEG:
            face.points[0] = make_float3(x0, y0, z0);
            face.points[1] = make_float3(x0, y0, z1);
            face.points[2] = make_float3(x0, y1, z1);
            face.points[3] = make_float3(x0, y1, z0);
            break;
        case FACE_X_POS:
            face.points[0] = make_float3(x1, y0, z0);
            face.points[1] = make_float3(x1, y1, z0);
            face.points[2] = make_float3(x1, y1, z1);
            face.points[3] = make_float3(x1, y0, z1);
            break;
        case FACE_Y_NEG:
            face.points[0] = make_float3(x0, y0, z0);
            face.points[1] = make_float3(x1, y0, z0);
            face.points[2] = make_float3(x1, y0, z1);
            face.points[3] = make_float3(x0, y0, z1);
            break;
        case FACE_Y_POS:
            face.points[0] = make_float3(x0, y1, z0);
            face.points[1] = make_float3(x0, y1, z1);
            face.points[2] = make_float3(x1, y1, z1);
            face.points[3] = make_float3(x1, y1, z0);
            break;
        case FACE_Z_NEG:
            face.points[0] = make_float3(x0, y0, z0);
            face.points[1] = make_float3(x0, y1, z0);
            face.points[2] = make_float3(x1, y1, z0);
            face.points[3] = make_float3(x1, y0, z0);
            break;
        default: // FACE_Z_POS
            face.points[0] = make_float3(x0, y0, z1);
            face.points[1] = make_float3(x1, y0, z1);
            face.points[2] = make_float3(x1, y1, z1);
            face.points[3] = make_float3(x0, y1, z1);
            break;
    }
}

__global__ void generateFacesKernel(const Cube* cubes, int n,
                                    const int* region_ids,
                                    const unsigned char* masks,
                                    const int* offsets,
                                    MergedFace* faces) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const Cube cube = cubes[tid];
    const unsigned char mask = masks[tid];
    int output = offsets[tid];
    for (int direction = 0; direction < 6; ++direction) {
        if ((mask & (1u << direction)) == 0) continue;
        MergedFace& face = faces[output++];
        setFacePoints(face, cube, direction);
        face.region_id = region_ids[tid];
        face.level = cube.level;
        face.state = cube.state;
        face.direction = direction;
    }
}

void launchBuildMergedSurface(const Cube* d_cubes, int n_cubes,
                              MergedFace** d_faces, int* n_faces,
                              int* n_regions) {
    *d_faces = nullptr;
    *n_faces = 0;
    *n_regions = 0;
    if (n_cubes <= 0) return;

    constexpr int threads = 256;
    const int blocks = (n_cubes + threads - 1) / threads;

    LeafKey* d_keys = nullptr;
    int* d_parent = nullptr;
    int* d_sorted_roots = nullptr;
    int* d_region_ids = nullptr;
    unsigned char* d_masks = nullptr;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;

    CUDA_CHECK(cudaMalloc(&d_keys, static_cast<size_t>(n_cubes) * sizeof(LeafKey)));
    CUDA_CHECK(cudaMalloc(&d_parent, static_cast<size_t>(n_cubes) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_roots, static_cast<size_t>(n_cubes) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_region_ids, static_cast<size_t>(n_cubes) * sizeof(int)));

    buildMergeLeafKeysKernel<<<blocks, threads>>>(d_cubes, n_cubes, d_keys);
    initializeParentsKernel<<<blocks, threads>>>(d_parent, n_cubes);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<LeafKey> key_begin(d_keys);
    thrust::sort(key_begin, key_begin + n_cubes, LeafKeyLess());

    unionSameLevelNeighborsKernel<<<blocks, threads>>>(d_cubes, n_cubes, d_keys,
                                                         d_parent);
    CUDA_CHECK(cudaGetLastError());

    // El árbol de padres siempre baja de índice. 32 saltos cubren cualquier
    // arreglo indexado por int y dejan la raíz de cada componente en un paso.
    for (int iteration = 0; iteration < 32; ++iteration)
        compressParentsKernel<<<blocks, threads>>>(d_parent, n_cubes);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(d_sorted_roots, d_parent,
                          static_cast<size_t>(n_cubes) * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    thrust::device_ptr<int> sorted_begin(d_sorted_roots);
    thrust::sort(sorted_begin, sorted_begin + n_cubes);
    const thrust::device_ptr<int> unique_end =
        thrust::unique(sorted_begin, sorted_begin + n_cubes);
    *n_regions = static_cast<int>(unique_end - sorted_begin);

    thrust::device_ptr<int> parent_begin(d_parent);
    thrust::device_ptr<int> region_begin(d_region_ids);
    thrust::lower_bound(sorted_begin, unique_end, parent_begin,
                        parent_begin + n_cubes, region_begin);

    CUDA_CHECK(cudaFree(d_parent));
    CUDA_CHECK(cudaFree(d_sorted_roots));

    CUDA_CHECK(cudaMalloc(&d_masks, static_cast<size_t>(n_cubes) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_counts, static_cast<size_t>(n_cubes) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, static_cast<size_t>(n_cubes) * sizeof(int)));
    markExteriorFacesKernel<<<blocks, threads>>>(d_cubes, n_cubes, d_keys,
                                                  d_masks, d_counts);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<int> count_begin(d_counts);
    thrust::device_ptr<int> offset_begin(d_offsets);
    thrust::exclusive_scan(count_begin, count_begin + n_cubes, offset_begin);

    int last_offset = 0;
    int last_count = 0;
    CUDA_CHECK(cudaMemcpy(&last_offset, d_offsets + n_cubes - 1, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_count, d_counts + n_cubes - 1, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (last_offset > INT_MAX - last_count) {
        fprintf(stderr, "Error: demasiadas caras para la salida VTK Legacy.\n");
        exit(EXIT_FAILURE);
    }
    *n_faces = last_offset + last_count;

    if (*n_faces > 0) {
        CUDA_CHECK(cudaMalloc(d_faces, static_cast<size_t>(*n_faces) * sizeof(MergedFace)));
        generateFacesKernel<<<blocks, threads>>>(d_cubes, n_cubes, d_region_ids,
                                                 d_masks, offset_begin.get(), *d_faces);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_region_ids));
    CUDA_CHECK(cudaFree(d_masks));
    CUDA_CHECK(cudaDeviceSynchronize());
}
