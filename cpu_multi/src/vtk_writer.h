#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include "common.h"

// Exporta los cubos a formato VTK Legacy ASCII (UNSTRUCTURED_GRID).
void writeVTK(const char* filename, const Cube* cubes, int n_cubes);

#endif // VTK_WRITER_H
