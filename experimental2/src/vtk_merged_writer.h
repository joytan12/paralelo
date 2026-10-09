#ifndef VTK_MERGED_WRITER_H
#define VTK_MERGED_WRITER_H

#include "merged_face.cuh"

// Escribe la superficie de las regiones unidas como VTK Legacy BINARY
// POLYDATA. Cada celda es una cara exterior cuadrilateral.
void writeMergedVTK(const char* filename, const MergedFace* faces, int n_faces,
                    int n_regions, int n_source_cubes);

#endif // VTK_MERGED_WRITER_H
