#ifndef OCTREE_H
#define OCTREE_H

#include "common.h"
#include <cstddef>

// Refinamiento adaptativo con balance 2:1. Se subdividen los BORDER y los
// vecinos más gruesos necesarios; los INSIDE se conservan como hojas salvo
// cuando deben refinarse para cerrar el balance.
void refineAdaptive(const Cube* input, int n_input, int max_level,
                    Cube** output, int* n_output,
                    int* n_refined_parents);

// Refina uniformemente: cada cubo → 8 hijos (se conserva por compatibilidad)
void refine(const Cube* input, int n_input, Cube** output, int* n_output);

// Poda: elimina cubos STATE_OUTSIDE (paralelo con OpenMP)
void prune(const Cube* input, int n_input, Cube** output, int* n_output);

// --------------------------------------------------------------------------
// Memoria auxiliar por hoja de entrada, para calcular el pico exacto.
//   refine : banderas de subdivision + tabla de desplazamientos
//   prune  : marcas de supervivencia + prefijo exclusivo
// --------------------------------------------------------------------------
size_t refineScratchBytesPerLeaf();
size_t pruneScratchBytesPerLeaf();

#endif // OCTREE_H
