#include "octree.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <vector>

// --------------------------------------------------------------------------
// Vecindad topológica: cara, arista o vértice.
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

static int markBalancedRefinements(const Cube* input, int n_input,
                                    int max_level,
                                    std::vector<unsigned char>& split) {
    split.assign(n_input, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_input; i++) {
        split[i] = (input[i].state == STATE_BORDER &&
                    input[i].level < max_level) ? 1 : 0;
    }

    // La propagación es intencionalmente secuencial: cada nueva marca puede
    // activar vecinos de una marca anterior. El trabajo de clasificación y
    // generación de hijos sigue siendo paralelo.
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < n_input; i++) {
            if (!split[i]) continue;

            for (int j = 0; j < n_input; j++) {
                if (split[j] || input[j].level >= input[i].level) continue;
                if (!cubesTouch(input[i], input[j])) continue;

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

void refineAdaptive(const Cube* input, int n_input, int max_level,
                    Cube** output, int* n_output,
                    int* n_refined_parents) {
    std::vector<unsigned char> split;
    const int n_split = markBalancedRefinements(input, n_input, max_level, split);
    *n_refined_parents = n_split;

    const int capacity = n_input + n_split * 7;
    *output = (Cube*)malloc(std::max(1, capacity) * sizeof(Cube));

    int* offsets = (int*)malloc((n_input + 1) * sizeof(int));
    offsets[0] = 0;
    for (int i = 0; i < n_input; i++) {
        offsets[i + 1] = offsets[i] + (split[i] ? 8 : 1);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_input; i++) {
        int pos = offsets[i];
        if (!split[i]) {
            (*output)[pos] = input[i];
            continue;
        }

        for (int child = 0; child < 8; child++) {
            (*output)[pos + child] = makeChild(input[i], child);
        }
    }

    const int out = offsets[n_input];
    free(offsets);
    *n_output = out;
}

// --------------------------------------------------------------------------
// Refinamiento paralelo (OpenMP): cada hilo procesa un rango de cubos.
// Como cada cubo genera exactamente 8 hijos en posiciones fijas,
// no hay condiciones de carrera → se puede paralelizar sin sincronización.
// --------------------------------------------------------------------------
void refine(const Cube* input, int n_input, Cube** output, int* n_output) {
    *n_output = n_input * 8;
    *output   = (Cube*)malloc((*n_output) * sizeof(Cube));

    #pragma omp parallel for schedule(static)
    for (int tid = 0; tid < n_input; tid++) {
        const Cube& parent = input[tid];
        float child_half   = parent.half_size * 0.5f;
        int   child_level  = parent.level + 1;
        int   base         = tid * 8;

        for (int i = 0; i < 8; i++) {
            float dx = (i & 1) ? child_half : -child_half;
            float dy = (i & 2) ? child_half : -child_half;
            float dz = (i & 4) ? child_half : -child_half;

            Cube child;
            child.center.x  = parent.center.x + dx;
            child.center.y  = parent.center.y + dy;
            child.center.z  = parent.center.z + dz;
            child.half_size = child_half;
            child.level     = child_level;
            child.state     = STATE_UNCLASSIFIED;

            (*output)[base + i] = child;
        }
    }
}

// --------------------------------------------------------------------------
// Poda paralela (OpenMP):
//   1) Fase 1 — marcar en paralelo cuáles cubos sobreviven.
//   2) Fase 2 — prefijo exclusivo (scan) para calcular posiciones.
//   3) Fase 3 — copiar en paralelo al arreglo de salida.
// --------------------------------------------------------------------------
void prune(const Cube* input, int n_input, Cube** output, int* n_output) {
    // Fase 1: marcar supervivientes
    int* keep = (int*)malloc(n_input * sizeof(int));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_input; i++) {
        keep[i] = (input[i].state != STATE_OUTSIDE) ? 1 : 0;
    }

    // Fase 2: prefix sum exclusivo (secuencial, es O(n) rápido)
    int* prefix = (int*)malloc((n_input + 1) * sizeof(int));
    prefix[0] = 0;
    for (int i = 0; i < n_input; i++) {
        prefix[i + 1] = prefix[i] + keep[i];
    }
    int count = prefix[n_input];

    // Fase 3: copiar supervivientes en paralelo
    *output   = (Cube*)malloc(count * sizeof(Cube));
    *n_output = count;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_input; i++) {
        if (keep[i]) {
            (*output)[prefix[i]] = input[i];
        }
    }

    free(keep);
    free(prefix);
}

// --------------------------------------------------------------------------
// Memoria auxiliar por hoja de entrada.
// --------------------------------------------------------------------------
size_t refineScratchBytesPerLeaf() {
    // std::vector<unsigned char> split  +  int* offsets
    return sizeof(unsigned char) + sizeof(int);
}

size_t pruneScratchBytesPerLeaf() {
    // int* keep  +  int* prefix
    return 2 * sizeof(int);
}
