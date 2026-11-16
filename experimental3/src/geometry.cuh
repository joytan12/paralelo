#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

#include "octree.cuh"

// --------------------------------------------------------------------------
// Estructura que define una esfera en el espacio 3D.
// Se usa como figura de prueba para la clasificación geométrica.
// --------------------------------------------------------------------------
struct Sphere {
    float3 center;
    float  radius;
};

// --------------------------------------------------------------------------
// Lanza el kernel de clasificación cubo-esfera.
//
// Para cada cubo en d_cubes, determina si está completamente dentro,
// completamente fuera, o en el borde de la esfera, y actualiza su campo
// 'state' in-place.
//
//   d_cubes  – arreglo de cubos en device (se modifica in-place)
//   n_cubes  – cantidad de cubos
//   sphere   – esfera contra la cual clasificar (pasada por valor)
// --------------------------------------------------------------------------
void launchClassification(Cube* d_cubes, int n_cubes, Sphere sphere);

#endif // GEOMETRY_CUH
