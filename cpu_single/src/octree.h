#ifndef OCTREE_H
#define OCTREE_H

#include "common.h"
#include <cstddef>

// --------------------------------------------------------------------------
// Refina adaptativamente una hoja por nivel.
//
// Se subdividen los cubos BORDER y los vecinos más gruesos necesarios para
// mantener un balance 2:1. Los cubos INSIDE que no sean necesarios para el
// balance se conservan como hojas.
//
// n_refined_parents devuelve cuántos cubos padre fueron subdivididos.
void refineAdaptive(const Cube* input, int n_input, int max_level,
                    Cube** output, int* n_output,
                    int* n_refined_parents);

// Refina uniformemente los cubos de entrada: cada cubo produce 8 hijos.
// Se conserva por compatibilidad con la API original.
//
//   input    – arreglo de cubos del nivel actual
//   n_input  – cantidad de cubos en input
//   output   – [out] puntero al nuevo arreglo de hijos (malloc interno)
//   n_output – [out] cantidad de cubos generados (= n_input * 8)
//
// El llamador es responsable de liberar *output con free().
// --------------------------------------------------------------------------
void refine(const Cube* input, int n_input, Cube** output, int* n_output);

// --------------------------------------------------------------------------
// Poda: elimina cubos con state == STATE_OUTSIDE.
// Conserva solo INSIDE y BORDER.
//
//   input    – arreglo de cubos clasificados
//   n_input  – cantidad de cubos
//   output   – [out] arreglo compactado (malloc interno)
//   n_output – [out] cantidad de cubos sobrevivientes
//
// El llamador es responsable de liberar *output con free().
// --------------------------------------------------------------------------
void prune(const Cube* input, int n_input, Cube** output, int* n_output);

// --------------------------------------------------------------------------
// Memoria auxiliar por hoja de entrada, para calcular el pico exacto.
//   refine : vector de banderas de subdivision
//   prune  : la version secuencial no usa auxiliares
// --------------------------------------------------------------------------
size_t refineScratchBytesPerLeaf();
size_t pruneScratchBytesPerLeaf();

#endif // OCTREE_H
