#include "mesh_classify.h"
#include <cmath>

// --------------------------------------------------------------------------
// Helpers inline
// --------------------------------------------------------------------------
static inline float fmin2(float a, float b) { return a < b ? a : b; }
static inline float fmax2(float a, float b) { return a > b ? a : b; }

// --------------------------------------------------------------------------
// Test de intersección triángulo-AABB usando el Separating Axis Theorem (SAT).
//
// Basado en: Akenine-Möller, "Fast 3D Triangle-Box Overlap Testing" (2001).
//
// El AABB está centrado en (cx,cy,cz) con semilado h.
// Devuelve true si hay intersección (borde), false si son disjuntos.
// --------------------------------------------------------------------------
static bool triangleAABBIntersect(
    Vec3 v0, Vec3 v1, Vec3 v2,
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

    // --- Test de los 9 ejes cruzados ---

    // a00 = (1,0,0) x e0
    p0 =  e0y * t0z - e0z * t0y;
    p2 =  e0y * t2z - e0z * t2y;
    r  = h * (fabsf(e0y) + fabsf(e0z));
    if (fmax2(-fmax2(p0,p2), fmin2(p0,p2)) > r) return false;

    // a01 = (1,0,0) x e1
    p0 =  e1y * t0z - e1z * t0y;
    p2 =  e1y * t2z - e1z * t2y;
    r  = h * (fabsf(e1y) + fabsf(e1z));
    if (fmax2(-fmax2(p0,p2), fmin2(p0,p2)) > r) return false;

    // a02 = (1,0,0) x e2
    p0 =  e2y * t0z - e2z * t0y;
    p1 =  e2y * t1z - e2z * t1y;
    r  = h * (fabsf(e2y) + fabsf(e2z));
    if (fmax2(-fmax2(p0,p1), fmin2(p0,p1)) > r) return false;

    // a10 = (0,1,0) x e0
    p0 =  e0z * t0x - e0x * t0z;
    p2 =  e0z * t2x - e0x * t2z;
    r  = h * (fabsf(e0z) + fabsf(e0x));
    if (fmax2(-fmax2(p0,p2), fmin2(p0,p2)) > r) return false;

    // a11 = (0,1,0) x e1
    p0 =  e1z * t0x - e1x * t0z;
    p2 =  e1z * t2x - e1x * t2z;
    r  = h * (fabsf(e1z) + fabsf(e1x));
    if (fmax2(-fmax2(p0,p2), fmin2(p0,p2)) > r) return false;

    // a12 = (0,1,0) x e2
    p0 =  e2z * t0x - e2x * t0z;
    p1 =  e2z * t1x - e2x * t1z;
    r  = h * (fabsf(e2z) + fabsf(e2x));
    if (fmax2(-fmax2(p0,p1), fmin2(p0,p1)) > r) return false;

    // a20 = (0,0,1) x e0
    p0 = -e0y * t0x + e0x * t0y;
    p2 = -e0y * t2x + e0x * t2y;
    r  = h * (fabsf(e0y) + fabsf(e0x));
    if (fmax2(-fmax2(p0,p2), fmin2(p0,p2)) > r) return false;

    // a21 = (0,0,1) x e1
    p0 = -e1y * t0x + e1x * t0y;
    p2 = -e1y * t2x + e1x * t2y;
    r  = h * (fabsf(e1y) + fabsf(e1x));
    if (fmax2(-fmax2(p0,p2), fmin2(p0,p2)) > r) return false;

    // a22 = (0,0,1) x e2
    p0 = -e2y * t0x + e2x * t0y;
    p1 = -e2y * t1x + e2x * t1y;
    r  = h * (fabsf(e2y) + fabsf(e2x));
    if (fmax2(-fmax2(p0,p1), fmin2(p0,p1)) > r) return false;

    // --- Test AABB vs AABB ---
    float minX = fmin2(fmin2(t0x, t1x), t2x);
    float maxX = fmax2(fmax2(t0x, t1x), t2x);
    if (minX >  h || maxX < -h) return false;

    float minY = fmin2(fmin2(t0y, t1y), t2y);
    float maxY = fmax2(fmax2(t0y, t1y), t2y);
    if (minY >  h || maxY < -h) return false;

    float minZ = fmin2(fmin2(t0z, t1z), t2z);
    float maxZ = fmax2(fmax2(t0z, t1z), t2z);
    if (minZ >  h || maxZ < -h) return false;

    // --- Test del plano del triángulo vs AABB ---
    float nx = e0y * e1z - e0z * e1y;
    float ny = e0z * e1x - e0x * e1z;
    float nz = e0x * e1y - e0y * e1x;
    float d  = nx * t0x + ny * t0y + nz * t0z;

    float radius = h * (fabsf(nx) + fabsf(ny) + fabsf(nz));
    if (fabsf(d) > radius) return false;

    return true;
}

// --------------------------------------------------------------------------
// Ray casting en +X desde el punto (ox, oy, oz).
// Número impar de intersecciones → punto dentro de la malla.
// --------------------------------------------------------------------------
static int rayCastX(float ox, float oy, float oz,
                    const Vec3* verts, const IVec3* tris, int n_tris)
{
    int count = 0;
    for (int i = 0; i < n_tris; i++) {
        IVec3 tri = tris[i];
        Vec3 v0 = verts[tri.x];
        Vec3 v1 = verts[tri.y];
        Vec3 v2 = verts[tri.z];

        float minY = fmin2(fmin2(v0.y, v1.y), v2.y);
        float maxY = fmax2(fmax2(v0.y, v1.y), v2.y);
        if (oy < minY || oy > maxY) continue;

        float minZ = fmin2(fmin2(v0.z, v1.z), v2.z);
        float maxZ = fmax2(fmax2(v0.z, v1.z), v2.z);
        if (oz < minZ || oz > maxZ) continue;

        // Möller–Trumbore, dirección +X = (1,0,0)
        float ex0 = v1.x - v0.x, ey0 = v1.y - v0.y, ez0 = v1.z - v0.z;
        float ex1 = v2.x - v0.x, ey1 = v2.y - v0.y, ez1 = v2.z - v0.z;

        float hx = 0.0f, hy = -ez1, hz = ey1;
        float a  = ex0 * hx + ey0 * hy + ez0 * hz;
        if (fabsf(a) < 1e-8f) continue;

        float f  = 1.0f / a;
        float sx = ox - v0.x, sy = oy - v0.y, sz = oz - v0.z;

        float u = f * (sx * hx + sy * hy + sz * hz);
        if (u < 0.0f || u > 1.0f) continue;

        float qx = sy * ez0 - sz * ey0;
        float qy = sz * ex0 - sx * ez0;
        float qz = sx * ey0 - sy * ex0;

        float v = f * qx;
        if (v < 0.0f || u + v > 1.0f) continue;

        float t = f * (ex1 * qx + ey1 * qy + ez1 * qz);
        if (t > 1e-8f) count++;
    }
    return count;
}

// --------------------------------------------------------------------------
// Clasificación secuencial (1 hilo): itera sobre todos los cubos.
// --------------------------------------------------------------------------
void classifyMesh(Cube*         cubes,
                  int           n_cubes,
                  const Vec3*   verts,
                  const IVec3*  tris,
                  int           n_tris)
{
    for (int tid = 0; tid < n_cubes; tid++) {
        float cx = cubes[tid].center.x;
        float cy = cubes[tid].center.y;
        float cz = cubes[tid].center.z;
        float h  = cubes[tid].half_size;

        // Paso 1: test SAT triángulo-AABB
        bool is_border = false;
        for (int i = 0; i < n_tris; i++) {
            IVec3 tri = tris[i];
            Vec3 v0 = verts[tri.x];
            Vec3 v1 = verts[tri.y];
            Vec3 v2 = verts[tri.z];

            if (triangleAABBIntersect(v0, v1, v2, cx, cy, cz, h)) {
                is_border = true;
                break;
            }
        }

        if (is_border) {
            cubes[tid].state = STATE_BORDER;
            continue;
        }

        // Paso 2: ray casting para determinar dentro/fuera
        int crossings = rayCastX(cx, cy, cz, verts, tris, n_tris);
        cubes[tid].state = (crossings % 2 == 1) ? STATE_INSIDE : STATE_OUTSIDE;
    }
}
