#include "mesh_classify.cuh"
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
// Helpers matemáticos __device__
// --------------------------------------------------------------------------

// Mínimo y máximo de dos floats
__device__ __forceinline__ float dmin(float a, float b) { return a < b ? a : b; }
__device__ __forceinline__ float dmax(float a, float b) { return a > b ? a : b; }

// --------------------------------------------------------------------------
// Test de intersección triángulo-AABB usando el Separating Axis Theorem (SAT).
//
// Basado en: Akenine-Möller, "Fast 3D Triangle-Box Overlap Testing" (2001).
//
// El AABB está centrado en (cx,cy,cz) con semilado h.
// El triángulo tiene vértices v0, v1, v2.
//
// Devuelve true si hay intersección (borde), false si son disjuntos.
// --------------------------------------------------------------------------
__device__ bool triangleAABBIntersect(
    float3 v0, float3 v1, float3 v2,
    float cx, float cy, float cz, float h)
{
    // Trasladar el triángulo al sistema centrado en el AABB
    float t0x = v0.x - cx, t0y = v0.y - cy, t0z = v0.z - cz;
    float t1x = v1.x - cx, t1y = v1.y - cy, t1z = v1.z - cz;
    float t2x = v2.x - cx, t2y = v2.y - cy, t2z = v2.z - cz;

    // Aristas del triángulo
    float e0x = t1x - t0x, e0y = t1y - t0y, e0z = t1z - t0z;
    float e1x = t2x - t1x, e1y = t2y - t1y, e1z = t2z - t1z;
    float e2x = t0x - t2x, e2y = t0y - t2y, e2z = t0z - t2z;

    float p0, p1, p2, r;

    // --- Test de los 9 ejes cruzados (arista x eje coord) ---

    // a00 = (1,0,0) x e0 = (0, -e0z, e0y)
    p0 =  e0y * t0z - e0z * t0y;
    p2 =  e0y * t2z - e0z * t2y;
    r  = h * (fabsf(e0y) + fabsf(e0z));
    if (dmax(-dmax(p0,p2), dmin(p0,p2)) > r) return false;

    // a01 = (1,0,0) x e1 = (0, -e1z, e1y)
    p0 =  e1y * t0z - e1z * t0y;
    p2 =  e1y * t2z - e1z * t2y;
    r  = h * (fabsf(e1y) + fabsf(e1z));
    if (dmax(-dmax(p0,p2), dmin(p0,p2)) > r) return false;

    // a02 = (1,0,0) x e2 = (0, -e2z, e2y)
    p0 =  e2y * t0z - e2z * t0y;
    p1 =  e2y * t1z - e2z * t1y;
    r  = h * (fabsf(e2y) + fabsf(e2z));
    if (dmax(-dmax(p0,p1), dmin(p0,p1)) > r) return false;

    // a10 = (0,1,0) x e0 = (e0z, 0, -e0x)
    p0 =  e0z * t0x - e0x * t0z;
    p2 =  e0z * t2x - e0x * t2z;
    r  = h * (fabsf(e0z) + fabsf(e0x));
    if (dmax(-dmax(p0,p2), dmin(p0,p2)) > r) return false;

    // a11 = (0,1,0) x e1 = (e1z, 0, -e1x)
    p0 =  e1z * t0x - e1x * t0z;
    p2 =  e1z * t2x - e1x * t2z;
    r  = h * (fabsf(e1z) + fabsf(e1x));
    if (dmax(-dmax(p0,p2), dmin(p0,p2)) > r) return false;

    // a12 = (0,1,0) x e2 = (e2z, 0, -e2x)
    p0 =  e2z * t0x - e2x * t0z;
    p1 =  e2z * t1x - e2x * t1z;
    r  = h * (fabsf(e2z) + fabsf(e2x));
    if (dmax(-dmax(p0,p1), dmin(p0,p1)) > r) return false;

    // a20 = (0,0,1) x e0 = (-e0y, e0x, 0)
    p0 = -e0y * t0x + e0x * t0y;
    p2 = -e0y * t2x + e0x * t2y;
    r  = h * (fabsf(e0y) + fabsf(e0x));
    if (dmax(-dmax(p0,p2), dmin(p0,p2)) > r) return false;

    // a21 = (0,0,1) x e1 = (-e1y, e1x, 0)
    p0 = -e1y * t0x + e1x * t0y;
    p2 = -e1y * t2x + e1x * t2y;
    r  = h * (fabsf(e1y) + fabsf(e1x));
    if (dmax(-dmax(p0,p2), dmin(p0,p2)) > r) return false;

    // a22 = (0,0,1) x e2 = (-e2y, e2x, 0)
    p0 = -e2y * t0x + e2x * t0y;
    p1 = -e2y * t1x + e2x * t1y;
    r  = h * (fabsf(e2y) + fabsf(e2x));
    if (dmax(-dmax(p0,p1), dmin(p0,p1)) > r) return false;

    // --- Test AABB vs AABB (proyecciones en los 3 ejes principales) ---
    float minX = dmin(dmin(t0x, t1x), t2x);
    float maxX = dmax(dmax(t0x, t1x), t2x);
    if (minX >  h || maxX < -h) return false;

    float minY = dmin(dmin(t0y, t1y), t2y);
    float maxY = dmax(dmax(t0y, t1y), t2y);
    if (minY >  h || maxY < -h) return false;

    float minZ = dmin(dmin(t0z, t1z), t2z);
    float maxZ = dmax(dmax(t0z, t1z), t2z);
    if (minZ >  h || maxZ < -h) return false;

    // --- Test del plano del triángulo vs AABB ---
    // Normal del triángulo = e0 x e1
    float nx = e0y * e1z - e0z * e1y;
    float ny = e0z * e1x - e0x * e1z;
    float nz = e0x * e1y - e0y * e1x;
    float d  = nx * t0x + ny * t0y + nz * t0z;

    // Radio efectivo del AABB proyectado sobre la normal
    float radius = h * (fabsf(nx) + fabsf(ny) + fabsf(nz));
    if (fabsf(d) > radius) return false;

    return true;  // No se encontró eje separador → intersección
}

// --------------------------------------------------------------------------
// Ray casting en +X desde el punto (ox, oy, oz).
// Cuenta cuántos triángulos cruza el rayo en la dirección +X.
// Número impar → punto dentro de la malla.
// --------------------------------------------------------------------------
__device__ int rayCastX(float ox, float oy, float oz,
                         const float3* verts, const int3* tris, int n_tris)
{
    int count = 0;
    for (int i = 0; i < n_tris; i++) {
        int3 tri = tris[i];
        float3 v0 = verts[tri.x];
        float3 v1 = verts[tri.y];
        float3 v2 = verts[tri.z];

        // El rayo es (ox + t, oy, oz) para t >= 0.
        // Solo interesa si el triángulo está al frente (t > 0).
        // Primero verificar que el rayo YZ pueda alcanzar el triángulo:
        float minY = dmin(dmin(v0.y, v1.y), v2.y);
        float maxY = dmax(dmax(v0.y, v1.y), v2.y);
        if (oy < minY || oy > maxY) continue;

        float minZ = dmin(dmin(v0.z, v1.z), v2.z);
        float maxZ = dmax(dmax(v0.z, v1.z), v2.z);
        if (oz < minZ || oz > maxZ) continue;

        // Intersección rayo-triángulo (Möller–Trumbore, dirección +X)
        // d = (1,0,0)
        float ex0 = v1.x - v0.x, ey0 = v1.y - v0.y, ez0 = v1.z - v0.z;
        float ex1 = v2.x - v0.x, ey1 = v2.y - v0.y, ez1 = v2.z - v0.z;

        // h = d x e1;  d=(1,0,0)  →  h = (0*ez1 - 0*ey1, 0*ex1 - 1*ez1, 1*ey1 - 0*ex1)
        //            = (0, -ez1, ey1)
        float hx = 0.0f, hy = -ez1, hz = ey1;

        // a = e0 · h
        float a = ex0 * hx + ey0 * hy + ez0 * hz;
        if (fabsf(a) < 1e-8f) continue;  // paralelo

        float f = 1.0f / a;
        float sx = ox - v0.x, sy = oy - v0.y, sz = oz - v0.z;

        // u = f * (s · h)
        float u = f * (sx * hx + sy * hy + sz * hz);
        if (u < 0.0f || u > 1.0f) continue;

        // q = s x e0;  q=(sy*ez0 - sz*ey0, sz*ex0 - sx*ez0, sx*ey0 - sy*ex0)
        float qx = sy * ez0 - sz * ey0;
        float qy = sz * ex0 - sx * ez0;
        float qz = sx * ey0 - sy * ex0;

        // v = f * (d · q);  d=(1,0,0) → v = f * qx
        float v = f * qx;
        if (v < 0.0f || u + v > 1.0f) continue;

        // t = f * (e1 · q)
        float t = f * (ex1 * qx + ey1 * qy + ez1 * qz);
        if (t > 1e-8f) count++;
    }
    return count;
}

// --------------------------------------------------------------------------
// Kernel principal: clasificar cada cubo contra la malla triangular.
// --------------------------------------------------------------------------
__global__ void classifyMeshKernel(Cube*         cubes,
                                    int           n_cubes,
                                    const float3* verts,
                                    const int3*   tris,
                                    int           n_tris)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cubes) return;

    Cube cube = cubes[tid];
    float cx = cube.center.x;
    float cy = cube.center.y;
    float cz = cube.center.z;
    float h  = cube.half_size;

    // Paso 1: test SAT triángulo-AABB
    for (int i = 0; i < n_tris; i++) {
        int3 tri = tris[i];
        float3 v0 = verts[tri.x];
        float3 v1 = verts[tri.y];
        float3 v2 = verts[tri.z];

        if (triangleAABBIntersect(v0, v1, v2, cx, cy, cz, h)) {
            cubes[tid].state = STATE_BORDER;
            return;
        }
    }

    // Paso 2: ray casting para determinar dentro/fuera
    int crossings = rayCastX(cx, cy, cz, verts, tris, n_tris);
    cubes[tid].state = (crossings % 2 == 1) ? STATE_INSIDE : STATE_OUTSIDE;
}

// --------------------------------------------------------------------------
// Función host: copia datos de malla al device y lanza el kernel.
// --------------------------------------------------------------------------
void launchClassificationMesh(Cube*         d_cubes,
                               int           n_cubes,
                               const float3* d_verts,
                               const int3*   d_tris,
                               int           n_tris)
{
    int threadsPerBlock = 256;
    int blocks = (n_cubes + threadsPerBlock - 1) / threadsPerBlock;

    classifyMeshKernel<<<blocks, threadsPerBlock>>>(
        d_cubes, n_cubes, d_verts, d_tris, n_tris);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
