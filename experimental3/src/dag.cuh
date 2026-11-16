#ifndef DAG_CUH
#define DAG_CUH

#include "octree.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <functional>

// --------------------------------------------------------------------------
// DAG con simetrías de espejo del octree final.
//
// El octree final es un arreglo plano de hojas. Aquí se guarda como un grafo
// acíclico dirigido: los subárboles idénticos, o idénticos salvo reflexión en
// X/Y/Z, se almacenan una sola vez. La geometría (centro, tamaño, grid_index)
// no se guarda: se deduce del camino que se recorre desde la raíz.
//
// Un nodo interno son 8 "slots" de 32 bits (uno por octante). El octante usa
// el mismo orden de bits que refineKernel / path_code en octree.cu:
//     octante = bit_x | (bit_y << 1) | (bit_z << 2)
//
// Slot de 32 bits:
//     bits 31..30  tipo   (DagTipo)
//     bits 29..27  máscara de espejo (bit 0 = X, bit 1 = Y, bit 2 = Z)
//     bits 26..0   índice en el arreglo de nodos únicos del nivel siguiente
//
// Semántica de una referencia NODO(idx, m): "el nodo idx visto a través de
// la reflexión m". Si C es el nodo guardado, la vista V es:
//     V[o] = reflejar_m( C[o ^ m] )
// donde reflejar_m sobre NODO(idx2, m2) da NODO(idx2, m2 ^ m) y deja igual
// las hojas y los vacíos. Las reflexiones forman un grupo con XOR, por lo que
// la máscara se acumula con XOR al descender.
// --------------------------------------------------------------------------

enum DagTipo {
    DAG_VACIO       = 0,   // subárbol podado (todo OUTSIDE o inexistente)
    DAG_HOJA_INSIDE = 1,   // coincide con STATE_INSIDE
    DAG_HOJA_BORDER = 2,   // coincide con STATE_BORDER
    DAG_NODO        = 3,   // referencia a un nodo interno del nivel siguiente
    DAG_FUERA       = 4    // solo lo devuelve dagLookup: fuera del dominio
};

#define DAG_BITS_IDX   27
#define DAG_MASK_IDX   ((1u << DAG_BITS_IDX) - 1u)
#define DAG_MAX_NIVEL  20                 // igual que el octree base
#define DAG_NIVELES    (DAG_MAX_NIVEL + 1)

struct DagNode {
    uint32_t child[8];
};

__host__ __device__ inline uint32_t dagSlot(uint32_t tipo, uint32_t mascara,
                                            uint32_t idx) {
    return (tipo << 30) | ((mascara & 7u) << DAG_BITS_IDX) |
           (idx & DAG_MASK_IDX);
}
__host__ __device__ inline uint32_t dagTipo(uint32_t s)    { return s >> 30; }
__host__ __device__ inline uint32_t dagMascara(uint32_t s) {
    return (s >> DAG_BITS_IDX) & 7u;
}
__host__ __device__ inline uint32_t dagIdx(uint32_t s)     { return s & DAG_MASK_IDX; }

// reflejar_m sobre un slot: solo los NODO acumulan la máscara.
__host__ __device__ inline uint32_t dagReflejar(uint32_t s, uint32_t m) {
    return dagTipo(s) == DAG_NODO ? (s ^ (m << DAG_BITS_IDX)) : s;
}

// --------------------------------------------------------------------------
// Vista del DAG para kernels (se pasa por valor). nodos[k] apunta a los
// nodos internos únicos de nivel k (hijos de nivel k+1). La raíz es un slot
// guardado en device.
// --------------------------------------------------------------------------
struct DagView {
    const uint32_t* d_raiz;
    const DagNode*  nodos[DAG_NIVELES];
    int             max_nivel;   // nivel de la hoja más profunda (-1 = vacío)
};

// Resultado de dagLookup: tipo (DagTipo) y nivel de la hoja que cubre la
// celda. Si la celda está subdividida (hay hojas más finas) el tipo es
// DAG_NODO y nivel es el nivel consultado.
struct DagHit {
    uint32_t tipo;
    int      nivel;
};

// Busca la celda (x, y, z) del nivel `nivel` descendiendo desde la raíz. En
// cada paso toma el octante de las coordenadas enteras, lo combina con la
// máscara acumulada (o ^ acum) y, si el slot es NODO, acumula su máscara.
__device__ __forceinline__ DagHit dagLookup(const DagView& v,
                                            int x, int y, int z, int nivel) {
    DagHit hit;
    if (nivel < 0 || nivel > DAG_MAX_NIVEL ||
        x < 0 || y < 0 || z < 0 ||
        x >= (1 << nivel) || y >= (1 << nivel) || z >= (1 << nivel)) {
        hit.tipo = DAG_FUERA;
        hit.nivel = -1;
        return hit;
    }

    uint32_t slot = *v.d_raiz;
    uint32_t acum = dagMascara(slot);       // máscara de la referencia raíz
    for (int k = 0; k < nivel; ++k) {
        if (dagTipo(slot) != DAG_NODO) {
            hit.tipo = dagTipo(slot);
            hit.nivel = k;                  // la hoja (o vacío) cubre la celda
            return hit;
        }
        const int bit = nivel - k - 1;
        const int o = ((x >> bit) & 1) | (((y >> bit) & 1) << 1) |
                      (((z >> bit) & 1) << 2);
        slot = v.nodos[k][dagIdx(slot)].child[o ^ acum];
        if (dagTipo(slot) == DAG_NODO) acum ^= dagMascara(slot);
    }
    hit.tipo = dagTipo(slot);
    hit.nivel = nivel;
    return hit;
}

// --------------------------------------------------------------------------
// DAG construido en device más sus estadísticas (contadores en el host).
// --------------------------------------------------------------------------
struct Dag {
    bool      espejo;                     // true = considera las 8 reflexiones
    int       max_nivel;                  // hoja más profunda (-1 = vacío)
    uint32_t* d_raiz;                     // slot raíz (device)
    DagNode*  d_nodos[DAG_NIVELES];       // nodos únicos por nivel (device)

    long long n_hojas[DAG_NIVELES];       // hojas por nivel
    long long n_logicos[DAG_NIVELES];     // nodos internos sin compartir
    long long n_unicos[DAG_NIVELES];      // nodos internos únicos

    double    segundos;                   // tiempo de construcción
    size_t    pico_calculado;             // buffers propios (sin hojas de entrada)
    size_t    pico_medido;                // cudaMemGetInfo, incluye hojas y demás
    bool      por_subarboles;             // true: tiempo y picos incluyen el refinamiento
};

// Construye el DAG a partir del arreglo final de hojas (en device). No toca
// el refinamiento: todo el cómputo es kernels CUDA y thrust sobre device; al
// host solo llegan contadores enteros. Con espejo=false solo se considera la
// reflexión identidad. Termina el programa si el índice de 27 bits no alcanza.
void dagBuild(const Cube* d_cubes, int n_cubes, bool espejo, Dag* dag);

// Libera la memoria de device del DAG.
void dagFree(Dag* dag);

// Vista para pasar a kernels.
DagView dagView(const Dag& dag);

// Totales para el resumen (nodos de todos los niveles).
long long dagTotalLogicos(const Dag& dag);
long long dagTotalUnicos(const Dag& dag);

// Validación (a): expande el DAG a hojas (path_code, nivel, estado), las
// ordena por path_code y las compara con las hojas originales ordenadas.
// Devuelve true si hay el mismo conteo y son iguales elemento a elemento.
bool dagVerificarDescompresion(const Dag& dag, const Cube* d_cubes,
                               int n_cubes, bool verboso);

// Consulta cada hoja original con dagLookup(grid_index, level): devuelve la
// cantidad de hojas cuyo tipo o nivel no coincide (0 = correcto).
long long dagVerificarConsultas(const Dag& dag, const Cube* d_cubes,
                                int n_cubes);

// Validación (b): para cada hoja del DAG, sus 26 vecinos del mismo nivel
// vía dagLookup. Cuenta los pares (hoja, vecino) en que el vecino lo cubre
// una hoja 2 o más niveles más gruesa. 0 = balance 2:1 correcto.
// Si se pasa n_hojas != nullptr, devuelve el total de hojas expandidas.
long long dagVerificarBalance(const Dag& dag, long long* n_hojas);

// Tabla por nivel (hojas, nodos lógicos, únicos sin espejo y con espejo) y
// resumen de tamaños. Cualquiera de los dos DAG puede ser nullptr.
void dagImprimirTabla(const Dag* sin_espejo, const Dag* con_espejo);

// --------------------------------------------------------------------------
// Hito 2a: DAG por subárboles.
// --------------------------------------------------------------------------

// Contabilidad de VRAM compartida con quien arma el DAG por partes: permite
// declarar bytes ajenos (malla, hojas del subárbol, buffers del refinamiento)
// para que el pico calculado cuente todo lo que vive a la vez.
void   dagTrackerReiniciar();
void   dagTrackerExterno(size_t bytes);   // bytes ajenos vivos desde ahora
void   dagTrackerNota(size_t bytes_extra);// candidato puntual de pico
size_t dagTrackerPico();                  // máximo de (propios + ajenos)
size_t dagTrackerVivo();                  // propios vivos ahora

// DAG global con índices estables y solo-agregar (ver dag_global.inc).
struct DagGlobal;
DagGlobal* dagGlobalCrear(bool espejo, long long cap_inicial = 1024);
void       dagGlobalLiberar(DagGlobal* g);
long long  dagGlobalCuenta(const DagGlobal* g, int nivel);

// Reserva las referencias (clave, slot) de n subárboles, en 0 = VACÍO.
void dagReservarRefs(int n, unsigned long long** d_claves, uint32_t** d_slots);

// Entradas (clave = path_code, slot de hoja) de un arreglo de hojas del
// mismo nivel; d_claves y d_slots deben tener n elementos.
void dagEntradasDeHojas(const Cube* d_hojas, int n,
                        unsigned long long* d_claves, uint32_t* d_slots);

// Comprime un subárbol (todas las hojas de d_hojas son del nivel
// nivel_hojas, cuelgan de una misma celda de nivel nivel_raiz) contra el DAG
// global y deja su referencia (clave = path_code de la celda, slot) en
// d_clave_out / d_slot_out (device, 1 elemento cada uno).
void dagGlobalInsertarSubarbol(DagGlobal* g, const Cube* d_hojas, int n,
                               int nivel_hojas, int nivel_raiz,
                               unsigned long long* d_clave_out,
                               uint32_t* d_slot_out);

// Fase final: con las referencias de todos los subárboles (nivel nivel_refs;
// toma posesión de los arreglos) sube los niveles nivel_refs..1 y deja el
// resultado en *dag. max_nivel es el nivel de la hoja más profunda. Los
// slots VACÍO se descartan. Después se puede liberar g con dagGlobalLiberar.
void dagGlobalCompletar(DagGlobal* g, int nivel_refs,
                        unsigned long long* d_claves, uint32_t* d_slots, int n,
                        int max_nivel, Dag* dag);

// Conteo de hojas INSIDE/BORDER, hojas por nivel y nodos lógicos por nivel
// directamente desde el DAG, propagando multiplicidades de arriba hacia
// abajo sin expandir. Rellena dag->n_hojas y dag->n_logicos.
void dagContarDesdeDag(Dag* dag, long long* n_inside, long long* n_border);

// Compara dos DAG: mismas hojas (path, nivel, estado) y mismos nodos únicos
// por nivel. Expande ambos completos (solo para niveles moderados).
bool dagCompararDags(const Dag& a, const Dag& b, bool verboso);

// Rehace las hojas del subárbol cuya raíz es *d_raiz (device) y las deja en
// *d_hojas (device, el llamador libera con cudaFree).
typedef std::function<void(const Cube* d_raiz, Cube** d_hojas, int* n_hojas)>
    DagRegenerarFn;

// Expande el DAG subárbol por subárbol (celdas del nivel nivel_bloque),
// consulta cada hoja con dagLookup y, si regenerar no es vacío, la compara
// con las hojas rehechas desde d_raices. Libera cada bloque antes del
// siguiente. Devuelve true si todo coincide; n_hojas_total (opcional) recibe
// el total de hojas recorridas.
bool dagVerificarPorBloques(const Dag& dag, int nivel_bloque,
                            const Cube* d_raices, int n_raices,
                            const DagRegenerarFn& regenerar, bool verboso,
                            long long* n_hojas_total);


// --------------------------------------------------------------------------
// Hito 2b: subárboles adaptativos.
// --------------------------------------------------------------------------

// Como dagGlobalInsertarSubarbol, pero las hojas del subárbol pueden estar en
// cualquier nivel > nivel_raiz (o ser la propia raíz: un único cubo).
void dagGlobalInsertarSubarbolGeneral(DagGlobal* g, const Cube* d_hojas, int n,
                                      int nivel_raiz,
                                      unsigned long long* d_clave_out,
                                      uint32_t* d_slot_out);

// Descarta una pasada: libera el DAG global y las referencias.
void dagGlobalDescartar(DagGlobal* g, unsigned long long* d_claves,
                        uint32_t* d_slots, int n);

// Balance 2:1 global y consultas dagLookup por bloques (celdas del nivel
// nivel_bloque): cada bloque se expande, sus hojas consultan los 26 vecinos
// sobre el DAG completo y se libera. Independiente del chequeo de cáscaras.
bool dagVerificarBloquesAdaptivo(const Dag& dag, int nivel_bloque, bool verboso,
                                 long long* n_hojas_total,
                                 long long* violaciones_out);


// --------------------------------------------------------------------------
// Hito 3: comparación por cobertura. Para el caso en que la fase A subdividió
// cubos INSIDE que el monolítico deja gruesos: las hojas BORDER deben ser
// idénticas, cada INSIDE de `sub` debe estar contenida en una INSIDE de `ref`
// y el volumen INSIDE total debe coincidir. Devuelve true si se cumple todo.
// --------------------------------------------------------------------------
bool dagCompararCobertura(const Dag& sub, const Dag& ref, bool verboso,
                          long long* border_difs, long long* inside_no_contenidas,
                          long long* vol_sub, long long* vol_ref);

#endif // DAG_CUH
