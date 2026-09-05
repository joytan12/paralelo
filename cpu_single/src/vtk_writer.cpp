#include "vtk_writer.h"
#include <cstdio>
#include <cstdlib>

// --------------------------------------------------------------------------
// Exporta los cubos a formato VTK Legacy ASCII (UNSTRUCTURED_GRID).
//
// Formato:
//   POINTS  –  8 vértices por cubo (n_cubes * 8 puntos)
//   CELLS   –  1 celda hexaédrica por cubo
//   CELL_TYPES – VTK_HEXAHEDRON (tipo 12) para cada celda
//   CELL_DATA  – campo "level" y "state" para colorear en ParaView
// --------------------------------------------------------------------------
void writeVTK(const char* filename, const Cube* cubes, int n_cubes) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: no se pudo abrir '%s' para escritura.\n", filename);
        exit(EXIT_FAILURE);
    }

    int n_points = n_cubes * 8;

    // ---- Encabezado VTK ----
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Octree refinement level output\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");

    // ---- POINTS ----
    fprintf(fp, "POINTS %d float\n", n_points);

    for (int i = 0; i < n_cubes; i++) {
        float cx = cubes[i].center.x;
        float cy = cubes[i].center.y;
        float cz = cubes[i].center.z;
        float h  = cubes[i].half_size;

        // 8 vértices en orden VTK_HEXAHEDRON
        // Cara inferior (z-)
        fprintf(fp, "%f %f %f\n", cx - h, cy - h, cz - h); // 0
        fprintf(fp, "%f %f %f\n", cx + h, cy - h, cz - h); // 1
        fprintf(fp, "%f %f %f\n", cx + h, cy + h, cz - h); // 2
        fprintf(fp, "%f %f %f\n", cx - h, cy + h, cz - h); // 3
        // Cara superior (z+)
        fprintf(fp, "%f %f %f\n", cx - h, cy - h, cz + h); // 4
        fprintf(fp, "%f %f %f\n", cx + h, cy - h, cz + h); // 5
        fprintf(fp, "%f %f %f\n", cx + h, cy + h, cz + h); // 6
        fprintf(fp, "%f %f %f\n", cx - h, cy + h, cz + h); // 7
    }

    // ---- CELLS ----
    fprintf(fp, "CELLS %d %d\n", n_cubes, n_cubes * 9);

    for (int i = 0; i < n_cubes; i++) {
        int base = i * 8;
        fprintf(fp, "8 %d %d %d %d %d %d %d %d\n",
                base + 0, base + 1, base + 2, base + 3,
                base + 4, base + 5, base + 6, base + 7);
    }

    // ---- CELL_TYPES ----
    fprintf(fp, "CELL_TYPES %d\n", n_cubes);
    for (int i = 0; i < n_cubes; i++) {
        fprintf(fp, "12\n"); // VTK_HEXAHEDRON
    }

    // ---- CELL_DATA ----
    fprintf(fp, "CELL_DATA %d\n", n_cubes);

    // Campo 1: nivel de refinamiento
    fprintf(fp, "SCALARS level int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int i = 0; i < n_cubes; i++) {
        fprintf(fp, "%d\n", cubes[i].level);
    }

    // Campo 2: estado de clasificación
    fprintf(fp, "SCALARS state int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int i = 0; i < n_cubes; i++) {
        fprintf(fp, "%d\n", cubes[i].state);
    }

    fclose(fp);
    printf("VTK exportado: %s  (%d cubos, %d puntos)\n",
           filename, n_cubes, n_points);
}
