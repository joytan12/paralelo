#ifndef SIMILITUD_CUH
#define SIMILITUD_CUH

#include "dag.cuh"

// --------------------------------------------------------------------------
// Hito 4: medición de similitud aproximada sobre el DAG con espejos.
//
// No cambia el DAG ni su formato: solo mide cuánto se ganaría si los nodos
// "casi iguales" se guardaran como diferencias (deltas) respecto de otro
// nodo. Todo el cómputo es kernels CUDA y thrust sobre device; al host solo
// llegan contadores e histogramas.
//
// Distancia entre dos nodos del mismo nivel = número de slots distintos
// (0 a 8), tomando el mínimo sobre las 8 reflexiones de uno de ellos. Dos
// referencias NODO equivalentes por el estabilizador del hijo cuentan como
// iguales.
//
// Medidas, por nivel L de los nodos comparados:
//   M1  distancia de cada nodo a su hermano más parecido (hijos NODO de un
//       mismo padre con índice distinto; los ya compartidos no cuentan).
//   M2  nodos con otro nodo del nivel a distancia 1, o a distancia 2, usando
//       solo las formas canónicas y usando las 8 vistas de cada nodo.
//   M3  tamaño estimado de un formato delta (cotas optimista y conservadora).
//   M4  similitud geométrica: cada nodo se expande a una rejilla de 8x8x8
//       celdas (3 niveles) con el estado de cada celda y se cuenta la
//       fracción de celdas que difieren del hermano más parecido.
//   M5  tamaño con nodos de tamaño variable (sin deltas), como línea base.
// --------------------------------------------------------------------------

struct SimNivel {
    long long n;             // nodos únicos del nivel

    // M1: [0] sin hermano comparable, [1] d=1, [2] d=2, [3] d=3, [4] d>=4.
    long long m1[5];
    long long m1_cero;       // distancia 0 entre nodos distintos (debe ser 0)

    // M2: nodos con vecino a distancia 1, y con vecino a distancia 2 pero no 1.
    long long d1_canon, d2_canon;     // solo formas canónicas
    long long d1_vistas, d2_vistas;   // con las 8 vistas de cada nodo

    // M3 conservadora: deltas de 1 slot con la mejor posición de slot del
    // nivel (en cada grupo, 1 base completa y el resto deltas).
    long long deltas_cons;

    // M4: [0] sin hermano comparable, [1] 0 %, [2] <=1 %, [3] <=5 %,
    //     [4] <=10 %, [5] <=25 %, [6] >25 % de las 512 celdas.
    long long m4[7];

    // M5: slots de tipo NODO del nivel.
    long long slots_nodo;
};

struct SimResultado {
    int      max_nivel;            // nivel de la hoja más profunda del DAG
    SimNivel niv[DAG_NIVELES];
    double   segundos;             // tiempo de la medición
    size_t   pico_medido;          // VRAM adicional máxima observada (driver)
};

// Costos supuestos del formato delta (bytes).
#define SIM_BYTES_NODO    32
#define SIM_BYTES_DELTA1   8
#define SIM_BYTES_DELTA2  16

// Calcula todas las medidas sobre un DAG construido con espejo.
void dagMedirSimilitud(const Dag& dag, SimResultado* r);

// Tamaños totales en bytes derivados de la medición.
long long simBytesActual(const SimResultado& r);
long long simBytesVariable(const SimResultado& r);      // M5
long long simBytesDeltaOptCanon(const SimResultado& r); // M3 optimista
long long simBytesDeltaOptVistas(const SimResultado& r);
long long simBytesDeltaCons(const SimResultado& r);     // M3 conservadora

// Imprime las tablas por nivel y el resumen de tamaños.
void dagImprimirSimilitud(const SimResultado& r);

#endif // SIMILITUD_CUH
