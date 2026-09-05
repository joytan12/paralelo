#include "octree.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// --------------------------------------------------------------------------
// Vecinos topológicos.
//
// Dos hojas son vecinas si sus AABB se tocan en al menos un eje y se
// superponen/tocan en los otros. Se incluyen cara, arista y vértice. Esta
// condición es deliberadamente conservadora para evitar discontinuidades
// topológicas en la malla adaptativa.
// --------------------------------------------------------------------------
static bool cubesTouch(const Cube& a, const Cube& b) {
    const float eps = 1e-5f * std::max(1.0f, std::max(a.half_size, b.half_size));

    const float a_min[3] = {
        a.center.x - a.half_size, a.center.y - a.half_size,
        a.center.z - a.half_size};
    const float a_max[3] = {
        a.center.x + a.half_size, a.center.y + a.half_size,
        a.center.z + a.half_size};
    const float b_min[3] = {
        b.center.x - b.half_size, b.center.y - b.half_size,
        b.center.z - b.half_size};
    const float b_max[3] = {
        b.center.x + b.half_size, b.center.y + b.half_size,
        b.center.z + b.half_size};

    bool overlap[3];
    bool boundary[3];
    for (int axis = 0; axis < 3; axis++) {
        overlap[axis] = a_min[axis] <= b_max[axis] + eps &&
                        b_min[axis] <= a_max[axis] + eps;
        boundary[axis] = std::fabs(a_max[axis] - b_min[axis]) <= eps ||
                         std::fabs(b_max[axis] - a_min[axis]) <= eps;
    }

    return overlap[0] && overlap[1] && overlap[2] &&
           (boundary[0] || boundary[1] || boundary[2]);
}

// --------------------------------------------------------------------------
// Marca los padres que deben subdividirse para refinar los BORDER y cerrar
// iterativamente el balance 2:1.
// --------------------------------------------------------------------------
static int markBalancedRefinements(const Cube* input, int n_input,
                                    int max_level,
                                    std::vector<unsigned char>& split) {
    split.assign(n_input, 0);

    for (int i = 0; i < n_input; i++) {
        split[i] = (input[i].state == STATE_BORDER &&
                    input[i].level < max_level) ? 1 : 0;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < n_input; i++) {
            if (!split[i]) continue;

            for (int j = 0; j < n_input; j++) {
                if (split[j] || input[j].level >= input[i].level) continue;
                if (!cubesTouch(input[i], input[j])) continue;

                // Si i se refina, un vecino j más grueso debe refinarse
                // también para que la diferencia final no supere un nivel.
                split[j] = 1;
                changed = true;
            }
        }
    }

    int count = 0;
    for (unsigned char value : split) count += value != 0;
    return count;
}

static Cube makeChild(const Cube& parent, int child_index) {
    const float child_half = parent.half_size * 0.5f;
    Cube child;
    child.center.x = parent.center.x + ((child_index & 1) ? child_half : -child_half);
    child.center.y = parent.center.y + ((child_index & 2) ? child_half : -child_half);
    child.center.z = parent.center.z + ((child_index & 4) ? child_half : -child_half);
    child.half_size = child_half;
    child.level = parent.level + 1;
    child.state = STATE_UNCLASSIFIED;
    return child;
}

// --------------------------------------------------------------------------
// Refinamiento adaptativo con balance 2:1.
// --------------------------------------------------------------------------
void refineAdaptive(const Cube* input, int n_input, int max_level,
                    Cube** output, int* n_output,
                    int* n_refined_parents) {
    std::vector<unsigned char> split;
    const int n_split = markBalancedRefinements(input, n_input, max_level, split);
    *n_refined_parents = n_split;

    const int capacity = n_input + n_split * 7;
    *output = (Cube*)malloc(std::max(1, capacity) * sizeof(Cube));

    int out = 0;
    for (int i = 0; i < n_input; i++) {
        if (!split[i]) {
            (*output)[out++] = input[i];
            continue;
        }

        for (int child = 0; child < 8; child++) {
            (*output)[out++] = makeChild(input[i], child);
        }
    }
    *n_output = out;
}

// --------------------------------------------------------------------------
// Refinamiento secuencial (1 hilo): cada cubo padre genera 8 hijos.
//
// Los 8 hijos se obtienen desplazando el centro del padre en ±(half_size/2)
// en cada eje (x, y, z), produciendo las 8 combinaciones posibles.
// El half_size de los hijos es la mitad del padre.
// --------------------------------------------------------------------------
void refine(const Cube* input, int n_input, Cube** output, int* n_output) {
    *n_output = n_input * 8;
    *output   = (Cube*)malloc((*n_output) * sizeof(Cube));

    for (int tid = 0; tid < n_input; tid++) {
        const Cube& parent   = input[tid];
        float child_half     = parent.half_size * 0.5f;
        int   child_level    = parent.level + 1;
        int   base           = tid * 8;

        // Las 8 combinaciones de desplazamiento (±1) en (x, y, z)
        for (int i = 0; i < 8; i++) {
            float dx = (i & 1) ? child_half : -child_half;
            float dy = (i & 2) ? child_half : -child_half;
            float dz = (i & 4) ? child_half : -child_half;

            Cube child;
            child.center.x = parent.center.x + dx;
            child.center.y = parent.center.y + dy;
            child.center.z = parent.center.z + dz;
            child.half_size = child_half;
            child.level     = child_level;
            child.state     = STATE_UNCLASSIFIED;

            (*output)[base + i] = child;
        }
    }
}

// --------------------------------------------------------------------------
// Poda secuencial: compacta el arreglo eliminando cubos STATE_OUTSIDE.
// --------------------------------------------------------------------------
void prune(const Cube* input, int n_input, Cube** output, int* n_output) {
    // Peor caso: todos sobreviven
    *output = (Cube*)malloc(n_input * sizeof(Cube));

    int count = 0;
    for (int i = 0; i < n_input; i++) {
        if (input[i].state != STATE_OUTSIDE) {
            (*output)[count++] = input[i];
        }
    }
    *n_output = count;
}

// --------------------------------------------------------------------------
// Memoria auxiliar por hoja de entrada.
// --------------------------------------------------------------------------
size_t refineScratchBytesPerLeaf() {
    // std::vector<unsigned char> split
    return sizeof(unsigned char);
}

size_t pruneScratchBytesPerLeaf() {
    // La poda secuencial compacta directamente, sin arreglos auxiliares.
    return 0;
}
