#ifndef OCTREE_CUH
#define OCTREE_CUH

#include <cuda_runtime.h>
#include <cstddef>

// --------------------------------------------------------------------------
// Estructura que representa un cubo en el octree.
// Identica a la del proyecto base: el formato en VRAM no cambia.
// --------------------------------------------------------------------------
struct Cube {
    float3 center;    // centro del cubo
    float  half_size; // medio lado
    int    level;     // nivel de refinamiento (0 = raiz)
    int    state;     // 0=fuera, 1=dentro, 2=borde, 3=sin clasificar

    // Coordenadas enteras del cubo dentro de su nivel. La raiz es (0,0,0)
    // y cada hijo duplica estas coordenadas y agrega un bit por eje.
    int3   grid_index;

    // Clave de camino Morton/base-8. La raiz vale 1 y cada hijo agrega su
    // octante en los 3 bits menos significativos.
    unsigned long long path_code;
};

// Estados de clasificacion compartidos por refinamiento y clasificacion.
enum CubeState {
    STATE_OUTSIDE      = 0,
    STATE_INSIDE       = 1,
    STATE_BORDER       = 2,
    STATE_UNCLASSIFIED = 3
};

// --------------------------------------------------------------------------
// Refinamiento uniforme: cada cubo de entrada produce 8 hijos.
// Se usa solo para bajar la raiz hasta el nivel de descomposicion.
// --------------------------------------------------------------------------
void launchRefinement(const Cube* d_input, int n_input,
                      Cube** d_output, int* n_output);

// --------------------------------------------------------------------------
// Refinamiento adaptativo con balance 2:1, con halo (ghosts) opcional.
//
// El marcado y el cierre del balance se ejecutan sobre la union
//     [ d_own (n_own) ] U [ d_ghost (n_ghost) ]
// pero solo se materializan hijos para la porcion propia. Los ghosts son
// hojas de OTROS subarboles, de solo lectura: participan en la busqueda de
// vecinos para que la regla 2:1 se cumpla a traves de la frontera, y nunca
// se copian ni se subdividen.
//
// d_ghost_split lleva, para cada ghost, la bandera de subdivision que su
// dueno calculo en ese mismo paso. Es imprescindible: una hoja de frontera
// puede estar marcada por el balance interno de su subarbol (por una celda
// fina que no vive en la cascara), y esa marca no se puede reconstruir desde
// fuera a partir de state y level.
//
// d_split_own devuelve la bandera de subdivision de cada hoja propia. Es lo
// que hay que guardar en la cascara para que los vecinos la vean. El llamador
// la libera con cudaFree; puede pasar nullptr si no le interesa.
//
// Con d_ghost == nullptr y n_ghost == 0 el comportamiento es identico al
// del proyecto base.
//
// Solo regresan al host contadores enteros; ningun Cube cruza el bus.
// --------------------------------------------------------------------------
void launchAdaptiveRefinement(const Cube* d_own, int n_own,
                              const Cube* d_ghost, int n_ghost,
                              const int* d_ghost_split,
                              int max_level,
                              Cube** d_output, int* n_output,
                              int* n_refined_parents,
                              int** d_split_own);

// --------------------------------------------------------------------------
// Poda (stream compaction): elimina los cubos con state == 0 (FUERA).
// Reserva el tamano exacto: primero cuenta en device, luego compacta.
// --------------------------------------------------------------------------
void launchPrune(const Cube* d_input, int n_input,
                 Cube** d_output, int* n_output);

// --------------------------------------------------------------------------
// Cuenta en device cuantos cubos tienen un estado dado.
// --------------------------------------------------------------------------
int launchCountByState(const Cube* d_input, int n_input, int target_state);

// --------------------------------------------------------------------------
// Cascara de un subarbol.
//
// Extrae las hojas que tocan la frontera exterior del subarbol al que
// pertenecen (su ancestro de nivel split_level). Son las unicas hojas que
// un subarbol vecino puede llegar a ver, y por eso son las unicas que hay
// que conservar como ghosts.
//
// Una hoja de nivel L dentro de un subarbol de nivel D esta en la cascara
// si alguna de sus coordenadas locales vale 0 o 2^(L-D)-1.
// --------------------------------------------------------------------------
// d_split acompana a d_input con la bandera de subdivision de cada hoja y se
// filtra con el mismo criterio, de modo que la cascara viaja completa: celda
// mas decision. Puede ser nullptr, y entonces las banderas salen en cero.
void launchExtractShell(const Cube* d_input, const int* d_split, int n_input,
                        int split_level,
                        Cube** d_shell, int** d_shell_split, int* n_shell);

// --------------------------------------------------------------------------
// Verificacion de la regla 2:1 sobre un conjunto de hojas.
//
// Para cada hoja se recorren sus 26 vecinos de su mismo nivel y se busca si
// el que los cubre es una hoja dos o mas niveles mas gruesa. Devuelve la
// cantidad de pares en violacion. Sobre la union de las cascaras finales
// comprueba exactamente lo que la descomposicion podria haber roto.
// --------------------------------------------------------------------------
unsigned long long launchVerifyBalance(const Cube* d_leaves, int n_leaves,
                                       int split_level);

// --------------------------------------------------------------------------
// Bytes auxiliares por hoja que reserva el refinamiento adaptativo.
// --------------------------------------------------------------------------
size_t adaptiveScratchBytesPerLeaf();

#endif // OCTREE_CUH
