#ifndef COMMON_H
#define COMMON_H

// =============================================================================
// common.h — Tipos base que reemplazan float3 / int3 de CUDA runtime.
// =============================================================================

#include <cmath>

// --------------------------------------------------------------------------
// Vector de 3 flotantes (equivalente a float3 de CUDA)
// --------------------------------------------------------------------------
struct Vec3 {
    float x, y, z;
};

inline Vec3 makeVec3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z; return v;
}

// --------------------------------------------------------------------------
// Vector de 3 enteros (equivalente a int3 de CUDA)
// --------------------------------------------------------------------------
struct IVec3 {
    int x, y, z;
};

inline IVec3 makeIVec3(int x, int y, int z) {
    IVec3 v; v.x = x; v.y = y; v.z = z; return v;
}

// --------------------------------------------------------------------------
// Estados de clasificación de un cubo
// --------------------------------------------------------------------------
enum CubeState {
    STATE_OUTSIDE      = 0,
    STATE_INSIDE       = 1,
    STATE_BORDER       = 2,
    STATE_UNCLASSIFIED = 3
};

// --------------------------------------------------------------------------
// Cubo en el octree (arreglo plano, CPU-friendly)
// --------------------------------------------------------------------------
struct Cube {
    Vec3  center;
    float half_size;
    int   level;
    int   state;
};

#endif // COMMON_H
