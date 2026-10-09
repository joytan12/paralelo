#ifndef MERGE_CUH
#define MERGE_CUH

#include "octree.cuh"
#include "merged_face.cuh"

// Agrupa en GPU las hojas que comparten una cara y tienen el mismo nivel.
// Genera solo las caras exteriores de cada región: las caras entre dos cubos
// de la misma región se eliminan. La memoria de d_faces pertenece al llamador.
void launchBuildMergedSurface(const Cube* d_cubes, int n_cubes,
                              MergedFace** d_faces, int* n_faces,
                              int* n_regions);

#endif // MERGE_CUH
