#ifndef OCTREE_CUH
#define OCTREE_CUH

#include <cuda_runtime.h>
#include <cstddef>

// --------------------------------------------------------------------------
// Estructura que representa un cubo en el octree.
// Se usa un arreglo plano (GPU-friendly) en vez de punteros tipo árbol.
// --------------------------------------------------------------------------
struct Cube {
    float3 center;    // centro del cubo
    float  half_size; // medio lado
    int    level;     // nivel de refinamiento (0 = raíz)
    int    state;     // 0=fuera, 1=dentro, 2=borde, 3=sin clasificar

    // Coordenadas enteras del cubo dentro de su nivel. La raíz es (0,0,0)
    // y cada hijo duplica estas coordenadas y agrega un bit por eje.
    int3   grid_index;

    // Clave de camino Morton/base-8. La raíz vale 1 y cada hijo agrega su
    // octante en los 3 bits menos significativos. Hasta nivel 20 cabe en 61
    // bits y permite localizar vecinos sin copiar el octree al host.
    unsigned long long path_code;
};

// Estados de clasificación compartidos por refinamiento y clasificación.
enum CubeState {
    STATE_OUTSIDE      = 0,
    STATE_INSIDE       = 1,
    STATE_BORDER       = 2,
    STATE_UNCLASSIFIED = 3
};

// --------------------------------------------------------------------------
// Lanza el kernel de refinamiento: cada cubo de entrada produce 8 hijos.
//
//   d_input   – arreglo de cubos del nivel actual (en device)
//   n_input   – cantidad de cubos en d_input
//   d_output  – [out] puntero al arreglo de hijos alojado en device
//   n_output  – [out] cantidad de cubos generados (= n_input * 8)
//
// La memoria de *d_output es asignada dentro de la función;
// el llamador es responsable de liberarla con cudaFree.
// --------------------------------------------------------------------------
void launchRefinement(const Cube* d_input, int n_input,
                      Cube** d_output, int* n_output);

// Refinamiento adaptativo con balance 2:1 completamente en device.
// Conserva las hojas existentes y subdivide los cubos BORDER junto con los
// vecinos más gruesos necesarios para mantener la diferencia de niveles.
// Solo se devuelven al host los contadores n_output y n_refined_parents.
void launchAdaptiveRefinement(const Cube* d_input, int n_input, int max_level,
                              Cube** d_output, int* n_output,
                              int* n_refined_parents);

// --------------------------------------------------------------------------
// Poda (stream compaction): elimina los cubos con state == 0 (FUERA).
// Conserva solo los cubos con state == 1 (DENTRO) o state == 2 (BORDE).
//
//   d_input   – arreglo de cubos clasificados (en device)
//   n_input   – cantidad de cubos en d_input
//   d_output  – [out] puntero al arreglo compactado en device
//   n_output  – [out] cantidad de cubos sobrevivientes
//
// La memoria de *d_output es asignada dentro de la función;
// el llamador es responsable de liberarla con cudaFree.
// --------------------------------------------------------------------------
void launchPrune(const Cube* d_input, int n_input,
                 Cube** d_output, int* n_output);

// --------------------------------------------------------------------------
// Filtra cubos por estado exacto (state == target_state).
// Útil para refinamiento adaptativo: separar INSIDE y BORDER.
//
//   d_input       – arreglo de cubos clasificados (en device)
//   n_input       – cantidad de cubos en d_input
//   target_state  – estado a conservar
//   d_output      – [out] arreglo filtrado en device
//   n_output      – [out] cantidad de cubos resultante
//
// La memoria de *d_output es asignada dentro de la función;
// el llamador es responsable de liberarla con cudaFree.
// --------------------------------------------------------------------------
void launchFilterByState(const Cube* d_input, int n_input,
                         int target_state,
                         Cube** d_output, int* n_output);

// --------------------------------------------------------------------------
// Cuenta en device cuantos cubos tienen un estado dado.
// Solo regresa al host el contador entero: los cubos permanecen en VRAM.
// --------------------------------------------------------------------------
int launchCountByState(const Cube* d_input, int n_input, int target_state);

// --------------------------------------------------------------------------
// Bytes auxiliares por hoja que reserva el refinamiento adaptativo:
// banderas de subdivision, prefijo exclusivo y claves de camino ordenadas.
// Se expone para poder calcular el pico exacto de VRAM por nivel.
// --------------------------------------------------------------------------
size_t adaptiveScratchBytesPerLeaf();

// --------------------------------------------------------------------------
// Hito 2b: adaptativo por subárboles con halo (portado de experimental/).
// Las funciones del refinamiento base de arriba no cambian.
// --------------------------------------------------------------------------

// Refinamiento adaptativo con balance 2:1 sobre [propias | ghosts]; solo se
// materializan hijos de las propias. d_ghost_split lleva la bandera de
// subdivisión que el dueño de cada ghost calculó en ese mismo paso.
// d_split_own devuelve la bandera de cada hoja propia (se libera con
// cudaFree; puede ser nullptr). Con n_ghost == 0 equivale al base.
void launchAdaptiveRefinementHalo(const Cube* d_own, int n_own,
                                  const Cube* d_ghost, int n_ghost,
                                  const int* d_ghost_split, int max_level,
                                  Cube** d_output, int* n_output,
                                  int* n_refined_parents, int** d_split_own);

// Poda in-place: compacta los sobrevivientes al frente de d_inout.
void launchPruneInPlace(Cube* d_inout, int n_input, int* n_output);

// Hojas de un subárbol que tocan su frontera exterior (con su bandera).
void launchExtractShell(const Cube* d_input, const int* d_split, int n_input,
                        int split_level,
                        Cube** d_shell, int** d_shell_split, int* n_shell);

// Pares (hoja, vecino) en violación de la regla 2:1 dentro de un conjunto
// de hojas (la unión de las cáscaras finales).
unsigned long long launchVerifyBalance(const Cube* d_leaves, int n_leaves,
                                       int split_level);

#endif // OCTREE_CUH
