#ifndef MESH_CLASSIFY_H
#define MESH_CLASSIFY_H

#include "common.h"

// --------------------------------------------------------------------------
// Clasifica cada cubo contra una malla triangular:
//   - Si algún triángulo intersecta el AABB → STATE_BORDER
//   - Si el centro del cubo tiene un número impar de intersecciones con
//     un rayo en +X → STATE_INSIDE
//   - De lo contrario → STATE_OUTSIDE
//
//   cubes   – arreglo de cubos (modificado in-place)
//   n_cubes – cantidad de cubos
//   verts   – vértices de la malla
//   tris    – triángulos de la malla (índices)
//   n_tris  – cantidad de triángulos
// --------------------------------------------------------------------------
void classifyMesh(Cube*         cubes,
                  int           n_cubes,
                  const Vec3*   verts,
                  const IVec3*  tris,
                  int           n_tris);

#endif // MESH_CLASSIFY_H
