#ifndef MESH_CLASSIFY_CUH
#define MESH_CLASSIFY_CUH

#include "octree.cuh"
#include "geometry.cuh"
#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// Lanza el kernel de clasificación cubo vs malla triangular.
//
// Estrategia por cubo (cada hilo maneja un cubo):
//   1) Test SAT triángulo-AABB: si algún triángulo intersecta el AABB del
//      cubo → STATE_BORDER.
//   2) Si ningún triángulo intersecta → ray casting en +X desde el centroide
//      del cubo para determinar si está dentro o fuera de la malla cerrada.
//      Número impar de cruces = DENTRO, par = FUERA.
//
// Parámetros:
//   d_cubes   – arreglo de cubos en device (se modifica in-place)
//   n_cubes   – cantidad de cubos
//   d_verts   – vértices de la malla en device
//   d_tris    – triángulos (int3: índices) de la malla en device
//   n_tris    – cantidad de triángulos
// --------------------------------------------------------------------------
void launchClassificationMesh(Cube*         d_cubes,
                               int           n_cubes,
                               const float3* d_verts,
                               const int3*   d_tris,
                               int           n_tris);

#endif // MESH_CLASSIFY_CUH
