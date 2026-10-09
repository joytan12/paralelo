#include "vtk_merged_writer.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void writeBE32(FILE* file, std::uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>((value >> 24) & 0xff),
        static_cast<unsigned char>((value >> 16) & 0xff),
        static_cast<unsigned char>((value >> 8) & 0xff),
        static_cast<unsigned char>(value & 0xff)};
    if (fwrite(bytes, sizeof(bytes), 1, file) != 1) {
        fprintf(stderr, "Error escribiendo datos VTK binarios.\n");
        exit(EXIT_FAILURE);
    }
}

static void writeBEInt(FILE* file, int value) {
    writeBE32(file, static_cast<std::uint32_t>(value));
}

static void writeBEFloat(FILE* file, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float debe tener 32 bits");
    std::memcpy(&bits, &value, sizeof(bits));
    writeBE32(file, bits);
}

void writeMergedVTK(const char* filename, const MergedFace* faces, int n_faces,
                    int n_regions, int n_source_cubes) {
    if (n_faces > INT_MAX / 5 || n_faces > INT_MAX / 4) {
        fprintf(stderr, "Error: la malla unida excede los límites VTK Legacy.\n");
        exit(EXIT_FAILURE);
    }

    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: no se pudo abrir '%s' para escritura.\n", filename);
        exit(EXIT_FAILURE);
    }

    const int n_points = n_faces * 4;
    fprintf(file, "# vtk DataFile Version 3.0\n");
    fprintf(file, "Merged same-level octree regions\n");
    fprintf(file, "BINARY\n");
    fprintf(file, "DATASET POLYDATA\n");
    fprintf(file, "POINTS %d float\n", n_points);
    for (int i = 0; i < n_faces; ++i) {
        for (int point = 0; point < 4; ++point) {
            const float3 p = faces[i].points[point];
            writeBEFloat(file, p.x);
            writeBEFloat(file, p.y);
            writeBEFloat(file, p.z);
        }
    }
    fputc('\n', file);

    fprintf(file, "POLYGONS %d %d\n", n_faces, n_faces * 5);
    for (int i = 0; i < n_faces; ++i) {
        const int first_point = i * 4;
        writeBEInt(file, 4);
        writeBEInt(file, first_point + 0);
        writeBEInt(file, first_point + 1);
        writeBEInt(file, first_point + 2);
        writeBEInt(file, first_point + 3);
    }
    fputc('\n', file);

    fprintf(file, "CELL_DATA %d\n", n_faces);
    fprintf(file, "SCALARS region_id int 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < n_faces; ++i) writeBEInt(file, faces[i].region_id);
    fputc('\n', file);

    fprintf(file, "SCALARS level int 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < n_faces; ++i) writeBEInt(file, faces[i].level);
    fputc('\n', file);

    fprintf(file, "SCALARS state int 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < n_faces; ++i) writeBEInt(file, faces[i].state);
    fputc('\n', file);

    fprintf(file, "SCALARS direction int 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < n_faces; ++i) writeBEInt(file, faces[i].direction);
    fputc('\n', file);

    fclose(file);
    printf("VTK de regiones unidas: %s  (%d cubos -> %d regiones, %d caras)\n",
           filename, n_source_cubes, n_regions, n_faces);
}
