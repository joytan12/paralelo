#ifndef DAG_SUBARBOLES_CUH
#define DAG_SUBARBOLES_CUH

#include "dag.cuh"
#include "octree.cuh"
#include <functional>

// --------------------------------------------------------------------------
// Hito 2a: DAG incremental por subárboles, modo uniforme.
//
// En el modo uniforme no hay búsqueda de vecinos, así que los subárboles son
// independientes: no hace falta halo ni pasadas. Flujo:
//
//   Fase A: refinar / clasificar / podar desde la raíz hasta el nivel de
//           descomposición D (split). Quedan <= 8^D raíces de subárbol.
//   Fase B: por cada raíz, refinar / clasificar / podar solo ese subárbol
//           hasta el nivel máximo, comprimirlo contra el DAG global y liberar
//           sus hojas antes de pasar al siguiente.
//   Fase C: construir los niveles D..1 y la raíz con las referencias
//           (idx, m) de los subárboles.
//
// El arreglo completo de hojas no existe nunca: el pico de VRAM es el del
// subárbol más grande más el DAG global.
// --------------------------------------------------------------------------

// Clasifica n cubos en device (malla o figura analítica).
typedef std::function<void(Cube* d_cubos, int n)> DagClasificarFn;

// Fases del perfil (--perfil), acumuladas por pasada en el adaptativo.
enum FasePerfil {
    PF_HALO = 0,      // armar el halo (copias de cáscaras vecinas)
    PF_REFINAR,       // refinamiento adaptativo con halo (marcado, ordenamiento, salida)
    PF_CASCARAS,      // extraer y guardar cáscaras (y liberar auxiliares)
    PF_CLASIFICAR,    // clasificar contra la malla / esfera
    PF_PODAR,         // poda in-place
    PF_INSERTAR,      // comprimir el subárbol contra el DAG global
    PF_VERIFICAR,     // verificación 2:1 de la unión de cáscaras
    PF_FASE_C,        // completar niveles D..1 y raíz
    PF_OTRO,          // resto (liberaciones, copias de hojas, descarte)
    PF_N
};
const char* nombreFasePerfil(int f);

struct SubarbolesConfig {
    int             max_nivel = 0;
    int             split = 1;          // nivel de descomposición D (1..max_nivel)
    bool            espejo = true;
    DagClasificarFn clasificar;
    size_t          bytes_malla = 0;    // VRAM constante de la malla
    bool            verboso = false;    // una línea por subárbol
    int             max_pasadas = 8;    // solo adaptativo: pasadas de cierre del balance
    bool            perfil = false;     // solo adaptativo: tiempo por fase y pasada (sincroniza el device)
};

struct SubarbolesInfo {
    Cube*  d_raices = nullptr;          // raíces de subárbol (device), nivel D
    int    n_raices = 0;
    int    n_subarboles = 0;            // no vacíos
    int    max_hojas_subarbol = 0;      // hojas del subárbol más grande
    double seg_fase_a = 0, seg_fase_b = 0, seg_fase_c = 0, seg_total = 0;
    size_t pico_calculado = 0;          // todo lo que vive a la vez (calculado)
    size_t pico_medido = 0;             // cudaMemGetInfo (stats.cu)
    // Solo adaptativo (hito 2b):
    int    pasadas = 0;                 // pasadas ejecutadas
    bool   convergio = true;            // la última pasada dio 0 violaciones
    long long violaciones_finales = 0;  // violaciones de la última pasada
    long long violaciones_pasada[16] = {};
    double seg_pasada[16] = {};
    size_t bytes_cascaras_max = 0;      // máximo de cáscaras residentes (halo previo + nuevo)
    size_t bytes_cascaras_final = 0;    // halo residente al terminar
    long long inside_subdivididos_fase_a = 0;  // cubos INSIDE subdivididos en la fase A
    double perfil[16][PF_N] = {};       // segundos por pasada y fase (si cfg.perfil)
};

// Construye el DAG por subárboles. Solo modo uniforme. Deja las raíces de la
// fase A en info->d_raices (para verificar); libérelas con subarbolesLiberar.
bool dagConstruirUniformePorSubarboles(const Cube& raiz_host,
                                       const SubarbolesConfig& cfg,
                                       Dag* dag, SubarbolesInfo* info);

// Hito 2b: construye el DAG adaptativo por subárboles con halo de cáscaras,
// una pasada tras otra (cada una con su DagGlobal) hasta 0 violaciones o
// cfg.max_pasadas. Si no converge deja el DAG de la última pasada y
// info->convergio = false.
bool dagConstruirAdaptivePorSubarboles(const Cube& raiz_host,
                                       const SubarbolesConfig& cfg,
                                       Dag* dag, SubarbolesInfo* info);

void subarbolesLiberar(SubarbolesInfo* info);

// Rehace las hojas del subárbol cuya raíz es *d_raiz (device) con el mismo
// refinamiento uniforme que la fase B. El llamador libera *d_hojas.
void subarbolRegenerar(const Cube* d_raiz, int max_nivel,
                       const DagClasificarFn& clasificar, size_t bytes_ajenos,
                       Cube** d_hojas, int* n_hojas);

// Un paso uniforme (refinar, clasificar, podar). Consume *d_actual.
void pasoUniforme(Cube** d_actual, int* n_actual,
                  const DagClasificarFn& clasificar, size_t bytes_ajenos);

#endif // DAG_SUBARBOLES_CUH
