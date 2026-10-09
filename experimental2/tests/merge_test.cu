#include "merge.cuh"

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err__));\
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

static unsigned long long pathCodeFromGridHost(int x, int y, int z, int level) {
    unsigned long long code = 1ULL;
    for (int bit = level - 1; bit >= 0; --bit) {
        const int octant = ((x >> bit) & 1) |
                           (((y >> bit) & 1) << 1) |
                           (((z >> bit) & 1) << 2);
        code = (code << 3) | static_cast<unsigned long long>(octant);
    }
    return code;
}

static Cube makeCube(int x, int y, int z, int level) {
    Cube cube{};
    const float h = 1.0f / static_cast<float>(1 << (level + 1));
    cube.center = make_float3(-0.5f + (2 * x + 1) * h,
                              -0.5f + (2 * y + 1) * h,
                              -0.5f + (2 * z + 1) * h);
    cube.half_size = h;
    cube.level = level;
    cube.state = STATE_INSIDE;
    cube.grid_index = make_int3(x, y, z);
    cube.path_code = pathCodeFromGridHost(x, y, z, level);
    return cube;
}

static bool verifyCase(const char* name, const Cube* host_cubes, int n_cubes,
                       int expected_regions, int expected_faces) {
    Cube* d_cubes = nullptr;
    MergedFace* d_faces = nullptr;
    int n_faces = 0;
    int n_regions = 0;
    cudaError_t err = cudaMalloc(&d_cubes, static_cast<size_t>(n_cubes) * sizeof(Cube));
    if (err != cudaSuccess) return false;
    err = cudaMemcpy(d_cubes, host_cubes, static_cast<size_t>(n_cubes) * sizeof(Cube),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return false;

    launchBuildMergedSurface(d_cubes, n_cubes, &d_faces, &n_faces, &n_regions);
    cudaFree(d_faces);
    cudaFree(d_cubes);

    if (n_regions != expected_regions || n_faces != expected_faces) {
        fprintf(stderr, "%s: esperado %d regiones y %d caras; obtenido %d y %d.\n",
                name, expected_regions, expected_faces, n_regions, n_faces);
        return false;
    }
    printf("%s: OK (%d regiones, %d caras)\n", name, n_regions, n_faces);
    return true;
}

int main() {
    CUDA_CHECK(cudaFree(0));

    // Bloque 2x2x1: cuatro cubos del nivel 1 comparten cuatro caras. Deben
    // formar una región y perder ocho caras internas: 4*6 - 2*4 = 16.
    const Cube block[] = {
        makeCube(0, 0, 0, 1), makeCube(1, 0, 0, 1),
        makeCube(0, 1, 0, 1), makeCube(1, 1, 0, 1)};
    if (!verifyCase("bloque mismo nivel", block, 4, 1, 16)) return EXIT_FAILURE;

    // Dos cubos que solo se tocan por arista no comparten región.
    const Cube edge_touch[] = {makeCube(0, 0, 0, 1), makeCube(1, 1, 0, 1)};
    if (!verifyCase("contacto por arista", edge_touch, 2, 2, 12)) return EXIT_FAILURE;

    // Un cubo de nivel 1 y uno de nivel 2 que se tocan no pueden unirse.
    const Cube mixed_level[] = {makeCube(0, 0, 0, 1), makeCube(2, 0, 0, 2)};
    if (!verifyCase("niveles distintos", mixed_level, 2, 2, 12)) return EXIT_FAILURE;

    printf("Todas las pruebas de unión GPU pasaron.\n");
    return EXIT_SUCCESS;
}
