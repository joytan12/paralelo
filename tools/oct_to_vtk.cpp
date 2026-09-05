// Convierte el checkpoint .oct de MixedOcTree (octree crudo, antes de los
// templates de transicion a malla mixta) a un VTK legacy de hexaedros,
// un cubo por octante, mismo estilo que el VTK que exporta el proyecto
// "paralelo" (8 puntos propios por cubo, sin compartir nodos).
//
// Formato .oct (Services.h / WriteOctreeMesh):
//   linea 1: np ne no
//   np lineas de puntos "x y z"
//   linea en blanco
//   ne lineas de aristas "a b c"
//   linea en blanco
//   no bloques de 2 lineas cada uno:
//     "8 i0 i1 i2 i3 i4 i5 i6 i7 level"
//     "n_caras_intersectadas [id...]"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <limits>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "uso: %s entrada.oct salida.vtk\n", argv[0]);
        return 1;
    }

    FILE* in = std::fopen(argv[1], "r");
    if (!in) { std::fprintf(stderr, "no pude abrir %s\n", argv[1]); return 1; }

    unsigned int np, ne, no;
    if (std::fscanf(in, "%u %u %u", &np, &ne, &no) != 3) {
        std::fprintf(stderr, "cabecera invalida\n"); return 1;
    }

    std::vector<double> px(np), py(np), pz(np);
    for (unsigned int i = 0; i < np; i++) {
        std::fscanf(in, "%lf %lf %lf", &px[i], &py[i], &pz[i]);
    }

    // aristas: "a b c" -> 3 tokens cada una
    for (unsigned int i = 0; i < ne; i++) {
        unsigned int a, b, c;
        std::fscanf(in, "%u %u %u", &a, &b, &c);
    }

    struct Octant {
        unsigned int idx[8];
        unsigned int level;
        unsigned int nfaces;
    };
    std::vector<Octant> octs(no);

    for (unsigned int i = 0; i < no; i++) {
        unsigned int n;
        std::fscanf(in, "%u", &n); // deberia ser 8
        for (unsigned int j = 0; j < 8 && j < n; j++) {
            std::fscanf(in, "%u", &octs[i].idx[j]);
        }
        std::fscanf(in, "%u", &octs[i].level);
        std::fscanf(in, "%u", &octs[i].nfaces);
        for (unsigned int j = 0; j < octs[i].nfaces; j++) {
            unsigned int dummy;
            std::fscanf(in, "%u", &dummy);
        }
    }
    std::fclose(in);

    FILE* out = std::fopen(argv[2], "w");
    if (!out) { std::fprintf(stderr, "no pude escribir %s\n", argv[2]); return 1; }

    std::fprintf(out, "# vtk DataFile Version 2.0\n");
    std::fprintf(out, "Octree puro (pre mixed-element) - MixedOcTree\n");
    std::fprintf(out, "ASCII\n");
    std::fprintf(out, "DATASET UNSTRUCTURED_GRID\n");

    unsigned int npts = no * 8;
    std::fprintf(out, "POINTS %u float\n", npts);

    // orden canonico VTK_HEXAHEDRON: base CCW (z-) luego tapa CCW (z+)
    for (unsigned int i = 0; i < no; i++) {
        double xmin = std::numeric_limits<double>::max(), xmax = -std::numeric_limits<double>::max();
        double ymin = xmin, ymax = xmax, zmin = xmin, zmax = xmax;
        for (unsigned int j = 0; j < 8; j++) {
            unsigned int id = octs[i].idx[j];
            if (px[id] < xmin) xmin = px[id];
            if (px[id] > xmax) xmax = px[id];
            if (py[id] < ymin) ymin = py[id];
            if (py[id] > ymax) ymax = py[id];
            if (pz[id] < zmin) zmin = pz[id];
            if (pz[id] > zmax) zmax = pz[id];
        }
        double cx[8] = {xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin};
        double cy[8] = {ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax};
        double cz[8] = {zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax};
        for (unsigned int j = 0; j < 8; j++) {
            std::fprintf(out, "%.8E %.8E %.8E\n", cx[j], cy[j], cz[j]);
        }
    }

    std::fprintf(out, "\nCELLS %u %u\n", no, no * 9);
    for (unsigned int i = 0; i < no; i++) {
        unsigned int base = i * 8;
        std::fprintf(out, "8 %u %u %u %u %u %u %u %u\n",
                     base, base+1, base+2, base+3, base+4, base+5, base+6, base+7);
    }

    std::fprintf(out, "\nCELL_TYPES %u\n", no);
    for (unsigned int i = 0; i < no; i++) std::fprintf(out, "12\n");

    std::fprintf(out, "\nCELL_DATA %u\n", no);
    std::fprintf(out, "SCALARS level int 1\nLOOKUP_TABLE default\n");
    for (unsigned int i = 0; i < no; i++) std::fprintf(out, "%u\n", octs[i].level);

    std::fprintf(out, "SCALARS border int 1\nLOOKUP_TABLE default\n");
    for (unsigned int i = 0; i < no; i++) std::fprintf(out, "%u\n", octs[i].nfaces > 0 ? 2 : 0);

    std::fclose(out);
    std::fprintf(stderr, "OK: %u cubos, %u puntos -> %s\n", no, npts, argv[2]);
    return 0;
}
