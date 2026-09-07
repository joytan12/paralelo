#ifndef VTK_STREAM_CUH
#define VTK_STREAM_CUH

#include "octree.cuh"

// --------------------------------------------------------------------------
// Escritor VTK por streaming desde VRAM.
//
// El proyecto base junta todos los cubos en un arreglo de host y despues
// escribe el archivo. Aqui no existe ese arreglo: cada subarbol terminado se
// evacua directamente desde VRAM.
//
// Reparto de trabajo:
//   GPU  – genera los bytes del archivo ya en big-endian (kernels por seccion)
//   DMA  – los baja por trozos a un anillo de dos buffers pinned
//   CPU  – solo hace fwrite de esos bytes; no calcula nada sobre los cubos
//
// El formato VTK Legacy es seccionado (POINTS, CELLS, CELL_TYPES, CELL_DATA)
// y los contadores van en la cabecera, que no se conoce hasta terminar. Por
// eso cada seccion se acumula en un archivo temporal y al cerrar se escribe
// la cabecera y se concatenan. El resultado es byte a byte el mismo formato
// que produce el escritor del proyecto base.
// --------------------------------------------------------------------------

struct VtkStream;

// Abre el escritor. chunk_cubes controla el tamano de los buffers: la seccion
// mas grande son 96 bytes por cubo, y se reservan dos buffers en device y dos
// pinned en host de chunk_cubes*96 bytes cada uno.
VtkStream* vtkStreamOpen(const char* out_path, int chunk_cubes);

// Descarta lo escrito y deja los temporales vacios, para repetir una pasada.
void vtkStreamReset(VtkStream* s);

// Evacua un subarbol terminado: n_cubes cubos que viven en VRAM.
void vtkStreamAppend(VtkStream* s, const Cube* d_cubes, int n_cubes);

// Cubos acumulados hasta ahora.
long long vtkStreamCount(const VtkStream* s);

// Escribe la cabecera, concatena las secciones y borra los temporales.
void vtkStreamFinish(VtkStream* s);

// Cierra sin producir el archivo final (borra los temporales).
void vtkStreamDiscard(VtkStream* s);

// Bytes que ocupara el archivo final con n_cubes cubos.
long long vtkEstimatedBytes(long long n_cubes);

// Limite de cubos que admite el formato: los indices de punto son int32 y
// hay 8 puntos por cubo.
long long vtkMaxCubes();

#endif // VTK_STREAM_CUH
