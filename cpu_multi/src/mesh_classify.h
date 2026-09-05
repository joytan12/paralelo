#ifndef MESH_CLASSIFY_H
#define MESH_CLASSIFY_H

#include "common.h"

// Clasifica cada cubo contra una malla triangular (OpenMP paralelo).
void classifyMesh(Cube*         cubes,
                  int           n_cubes,
                  const Vec3*   verts,
                  const IVec3*  tris,
                  int           n_tris);

#endif // MESH_CLASSIFY_H
