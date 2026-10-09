#include "geometry.cuh"
#include <cstdio>
#include <cstdlib>

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
// Kernel de clasificación cubo vs esfera.
//
// Algoritmo (prueba analítica precisa):
//
// 1) Calcular el punto más cercano del cubo (AABB) al centro de la esfera.
//    Si ese punto está fuera de la esfera → el cubo completo está FUERA.
//
// 2) Calcular el punto más lejano del cubo al centro de la esfera.
//    Si ese punto está dentro de la esfera → el cubo completo está DENTRO.
//
// 3) Si no se cumple ninguna de las dos → el cubo está en el BORDE
//    (la superficie de la esfera lo atraviesa).
// --------------------------------------------------------------------------
__global__ void classifySphereKernel(Cube* cubes, int n, Sphere sphere) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Cube cube = cubes[tid];

    float cx = cube.center.x;
    float cy = cube.center.y;
    float cz = cube.center.z;
    float h  = cube.half_size;

    float sx = sphere.center.x;
    float sy = sphere.center.y;
    float sz = sphere.center.z;
    float r  = sphere.radius;
    float r2 = r * r;

    // --- Punto más cercano del AABB al centro de la esfera ---
    // (clamping del centro de la esfera a los límites del cubo)
    float closest_x = fmaxf(cx - h, fminf(sx, cx + h));
    float closest_y = fmaxf(cy - h, fminf(sy, cy + h));
    float closest_z = fmaxf(cz - h, fminf(sz, cz + h));

    float dx_c = closest_x - sx;
    float dy_c = closest_y - sy;
    float dz_c = closest_z - sz;
    float dist_closest_sq = dx_c * dx_c + dy_c * dy_c + dz_c * dz_c;

    if (dist_closest_sq > r2) {
        // El punto más cercano del cubo está fuera de la esfera
        // → el cubo entero está FUERA
        cubes[tid].state = STATE_OUTSIDE;
        return;
    }

    // --- Punto más lejano del AABB al centro de la esfera ---
    // Para cada eje, elegir la cara del cubo más alejada del centro de la esfera
    float far_x = (fabsf((cx - h) - sx) > fabsf((cx + h) - sx))
                   ? (cx - h) : (cx + h);
    float far_y = (fabsf((cy - h) - sy) > fabsf((cy + h) - sy))
                   ? (cy - h) : (cy + h);
    float far_z = (fabsf((cz - h) - sz) > fabsf((cz + h) - sz))
                   ? (cz - h) : (cz + h);

    float dx_f = far_x - sx;
    float dy_f = far_y - sy;
    float dz_f = far_z - sz;
    float dist_farthest_sq = dx_f * dx_f + dy_f * dy_f + dz_f * dz_f;

    if (dist_farthest_sq <= r2) {
        // El punto más lejano del cubo está dentro de la esfera
        // → el cubo entero está DENTRO
        cubes[tid].state = STATE_INSIDE;
        return;
    }

    // --- Caso intermedio: la superficie de la esfera atraviesa el cubo ---
    cubes[tid].state = STATE_BORDER;
}

// --------------------------------------------------------------------------
// Función host: configura y lanza el kernel de clasificación.
// --------------------------------------------------------------------------
void launchClassification(Cube* d_cubes, int n_cubes, Sphere sphere) {
    int threadsPerBlock = 256;
    int blocks = (n_cubes + threadsPerBlock - 1) / threadsPerBlock;

    classifySphereKernel<<<blocks, threadsPerBlock>>>(d_cubes, n_cubes, sphere);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
