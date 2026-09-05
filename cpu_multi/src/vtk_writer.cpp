#include "vtk_writer.h"
#include <cstdio>
#include <cstdlib>

void writeVTK(const char* filename, const Cube* cubes, int n_cubes) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: no se pudo abrir '%s' para escritura.\n", filename);
        exit(EXIT_FAILURE);
    }

    int n_points = n_cubes * 8;

    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Octree refinement level output\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %d float\n", n_points);

    for (int i = 0; i < n_cubes; i++) {
        float cx = cubes[i].center.x;
        float cy = cubes[i].center.y;
        float cz = cubes[i].center.z;
        float h  = cubes[i].half_size;

        fprintf(fp, "%f %f %f\n", cx - h, cy - h, cz - h);
        fprintf(fp, "%f %f %f\n", cx + h, cy - h, cz - h);
        fprintf(fp, "%f %f %f\n", cx + h, cy + h, cz - h);
        fprintf(fp, "%f %f %f\n", cx - h, cy + h, cz - h);
        fprintf(fp, "%f %f %f\n", cx - h, cy - h, cz + h);
        fprintf(fp, "%f %f %f\n", cx + h, cy - h, cz + h);
        fprintf(fp, "%f %f %f\n", cx + h, cy + h, cz + h);
        fprintf(fp, "%f %f %f\n", cx - h, cy + h, cz + h);
    }

    fprintf(fp, "CELLS %d %d\n", n_cubes, n_cubes * 9);
    for (int i = 0; i < n_cubes; i++) {
        int base = i * 8;
        fprintf(fp, "8 %d %d %d %d %d %d %d %d\n",
                base+0, base+1, base+2, base+3,
                base+4, base+5, base+6, base+7);
    }

    fprintf(fp, "CELL_TYPES %d\n", n_cubes);
    for (int i = 0; i < n_cubes; i++) fprintf(fp, "12\n");

    fprintf(fp, "CELL_DATA %d\n", n_cubes);
    fprintf(fp, "SCALARS level int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int i = 0; i < n_cubes; i++) fprintf(fp, "%d\n", cubes[i].level);

    fprintf(fp, "SCALARS state int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int i = 0; i < n_cubes; i++) fprintf(fp, "%d\n", cubes[i].state);

    fclose(fp);
    printf("VTK exportado: %s  (%d cubos, %d puntos)\n", filename, n_cubes, n_points);
}
