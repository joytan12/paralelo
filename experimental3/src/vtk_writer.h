#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include "octree.cuh"

// --------------------------------------------------------------------------
// Escribe un archivo VTK Legacy BINARY (UNSTRUCTURED_GRID) con los cubos.
// Cada cubo se representa como un VTK_HEXAHEDRON (8 vértices).
// Incluye el campo "level" como CELL_DATA para colorear en ParaView.
//
//   filename – ruta del archivo de salida (ej. "output/octree_level_3.vtk")
//   cubes    – arreglo de cubos en memoria de HOST
//   n_cubes  – cantidad de cubos
// --------------------------------------------------------------------------
void writeVTK(const char* filename, const Cube* cubes, int n_cubes);

#endif // VTK_WRITER_H
