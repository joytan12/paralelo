#ifndef MERGED_FACE_CUH
#define MERGED_FACE_CUH

#include <cuda_runtime.h>

// Una cara exterior de una región formada por hojas del octree. Las cuatro
// posiciones están en orden antihorario vistas desde fuera de la región.
struct MergedFace {
    float3 points[4];
    int    region_id;
    int    level;
    int    state;
    int    direction;
};

enum MergedFaceDirection {
    FACE_X_NEG = 0,
    FACE_X_POS = 1,
    FACE_Y_NEG = 2,
    FACE_Y_POS = 3,
    FACE_Z_NEG = 4,
    FACE_Z_POS = 5
};

#endif // MERGED_FACE_CUH
