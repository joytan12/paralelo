#include "vtk_writer.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// --------------------------------------------------------------------------
// Exporta los cubos a formato VTK Legacy BINARY (UNSTRUCTURED_GRID).
//
// Formato:
//   POINTS  –  8 vértices por cubo (n_cubes * 8 puntos)
//   CELLS   –  1 celda hexaédrica por cubo
//   CELL_TYPES – VTK_HEXAHEDRON (tipo 12) para cada celda
//   CELL_DATA  – campo "level" (entero) para colorear en ParaView
//
// Orden de vértices del hexaedro VTK (vista desde arriba, z+ arriba):
//
//        6 -------- 7          Cara inferior (z-): 0 1 2 3
//       /|         /|          Cara superior (z+): 4 5 6 7
//      4 -------- 5 |
//      | 2 -------| 3
//      |/         |/
//      0 -------- 1
//
// --------------------------------------------------------------------------

// VTK Legacy BINARY exige que los datos numéricos estén en big-endian,
// independientemente de la arquitectura de la máquina que los escribe.
static void writeBE32(FILE* fp, std::uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>((value >> 24) & 0xff),
        static_cast<unsigned char>((value >> 16) & 0xff),
        static_cast<unsigned char>((value >> 8) & 0xff),
        static_cast<unsigned char>(value & 0xff)};
    if (fwrite(bytes, sizeof(bytes), 1, fp) != 1) {
        fprintf(stderr, "Error escribiendo datos VTK binarios.\n");
        exit(EXIT_FAILURE);
    }
}

static void writeBEInt(FILE* fp, int value) {
    writeBE32(fp, static_cast<std::uint32_t>(value));
}

static void writeBEFloat(FILE* fp, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float debe tener 32 bits");
    std::memcpy(&bits, &value, sizeof(bits));
    writeBE32(fp, bits);
}

void writeVTK(const char* filename, const Cube* cubes, int n_cubes) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: no se pudo abrir '%s' para escritura.\n", filename);
        exit(EXIT_FAILURE);
    }

    int n_points = n_cubes * 8;

    // ---- Encabezado VTK ----
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Octree refinement level output\n");
    fprintf(fp, "BINARY\n");
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
        writeBEFloat(fp, cx - h); writeBEFloat(fp, cy - h); writeBEFloat(fp, cz - h); // 0
        writeBEFloat(fp, cx + h); writeBEFloat(fp, cy - h); writeBEFloat(fp, cz - h); // 1
        writeBEFloat(fp, cx + h); writeBEFloat(fp, cy + h); writeBEFloat(fp, cz - h); // 2
        writeBEFloat(fp, cx - h); writeBEFloat(fp, cy + h); writeBEFloat(fp, cz - h); // 3
        // Cara superior (z+)
        writeBEFloat(fp, cx - h); writeBEFloat(fp, cy - h); writeBEFloat(fp, cz + h); // 4
        writeBEFloat(fp, cx + h); writeBEFloat(fp, cy - h); writeBEFloat(fp, cz + h); // 5
        writeBEFloat(fp, cx + h); writeBEFloat(fp, cy + h); writeBEFloat(fp, cz + h); // 6
        writeBEFloat(fp, cx - h); writeBEFloat(fp, cy + h); writeBEFloat(fp, cz + h); // 7
    }
    fputc('\n', fp);

    // ---- CELLS ----
    // Cada celda tiene 8 vértices, y la línea empieza con "8" (nº de puntos)
    // Tamaño total de la lista de celdas = n_cubes * (1 + 8) = n_cubes * 9
    fprintf(fp, "CELLS %d %d\n", n_cubes, n_cubes * 9);

    for (int i = 0; i < n_cubes; i++) {
        int base = i * 8;
        writeBEInt(fp, 8);
        for (int j = 0; j < 8; ++j)
            writeBEInt(fp, base + j);
    }
    fputc('\n', fp);

    // ---- CELL_TYPES ----
    fprintf(fp, "CELL_TYPES %d\n", n_cubes);
    for (int i = 0; i < n_cubes; i++) {
        writeBEInt(fp, 12); // VTK_HEXAHEDRON
    }
    fputc('\n', fp);

    // ---- CELL_DATA ----
    fprintf(fp, "CELL_DATA %d\n", n_cubes);

    // Campo 1: nivel de refinamiento
    fprintf(fp, "SCALARS level int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int i = 0; i < n_cubes; i++) {
        writeBEInt(fp, cubes[i].level);
    }
    fputc('\n', fp);

    // Campo 2: estado de clasificación (0=fuera, 1=dentro, 2=borde, 3=sin clasificar)
    fprintf(fp, "SCALARS state int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int i = 0; i < n_cubes; i++) {
        writeBEInt(fp, cubes[i].state);
    }
    fputc('\n', fp);

    fclose(fp);
    printf("VTK binario exportado: %s  (%d cubos, %d puntos)\n",
           filename, n_cubes, n_points);
}
