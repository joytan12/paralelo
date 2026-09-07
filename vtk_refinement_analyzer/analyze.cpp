#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

struct Cell { float xmin, xmax, ymin, ymax, zmin, zmax; int level = -1; };

static bool readLine(FILE* file, std::string& value) {
    char buffer[256];
    if (!std::fgets(buffer, sizeof(buffer), file)) return false;
    value = buffer;
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) value.pop_back();
    return true;
}

static std::uint32_t readBE32(FILE* file) {
    unsigned char bytes[4];
    if (std::fread(bytes, 1, 4, file) != 4) {
        std::fprintf(stderr, "Error: VTK truncado al leer datos binarios.\n");
        std::exit(EXIT_FAILURE);
    }
    return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16) |
           (std::uint32_t(bytes[2]) << 8) | std::uint32_t(bytes[3]);
}

static float readBEFloat(FILE* file) {
    std::uint32_t bits = readBE32(file);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static int readBEInt(FILE* file) { return static_cast<std::int32_t>(readBE32(file)); }

static bool touches(const Cell& a, const Cell& b) {
    const float eps = 1e-6f;
    return a.xmin <= b.xmax + eps && b.xmin <= a.xmax + eps &&
           a.ymin <= b.ymax + eps && b.ymin <= a.ymax + eps &&
           a.zmin <= b.zmax + eps && b.zmin <= a.zmax + eps;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Uso: %s output/archivo.vtk\n", argv[0]);
        return 1;
    }
    FILE* file = std::fopen(argv[1], "rb");
    if (!file) { std::perror(argv[1]); return 1; }
    std::string text;
    if (!readLine(file, text) || text != "# vtk DataFile Version 3.0") return 1;
    readLine(file, text);
    if (!readLine(file, text) || text != "BINARY") {
        std::fprintf(stderr, "Error: el analizador requiere VTK Legacy BINARY.\n"); return 1;
    }
    if (!readLine(file, text) || text != "DATASET UNSTRUCTURED_GRID") return 1;

    int n_points = 0;
    if (!readLine(file, text) || std::sscanf(text.c_str(), "POINTS %d float", &n_points) != 1) return 1;
    std::vector<float> points(static_cast<size_t>(n_points) * 3);
    for (float& value : points) value = readBEFloat(file);
    std::fgetc(file);

    int n_cells = 0, cell_values = 0;
    if (!readLine(file, text) || std::sscanf(text.c_str(), "CELLS %d %d", &n_cells, &cell_values) != 2) return 1;
    std::vector<Cell> cells(static_cast<size_t>(n_cells));
    for (Cell& cell : cells) {
        if (readBEInt(file) != 8) return 1;
        cell.xmin = cell.ymin = cell.zmin = std::numeric_limits<float>::max();
        cell.xmax = cell.ymax = cell.zmax = -std::numeric_limits<float>::max();
        for (int j = 0; j < 8; ++j) {
            int point = readBEInt(file);
            if (point < 0 || point >= n_points) return 1;
            cell.xmin = std::min(cell.xmin, points[point * 3]);
            cell.xmax = std::max(cell.xmax, points[point * 3]);
            cell.ymin = std::min(cell.ymin, points[point * 3 + 1]);
            cell.ymax = std::max(cell.ymax, points[point * 3 + 1]);
            cell.zmin = std::min(cell.zmin, points[point * 3 + 2]);
            cell.zmax = std::max(cell.zmax, points[point * 3 + 2]);
        }
    }
    std::fgetc(file);

    int n_types = 0;
    if (!readLine(file, text) || std::sscanf(text.c_str(), "CELL_TYPES %d", &n_types) != 1 || n_types != n_cells) return 1;
    for (int i = 0; i < n_cells; ++i) readBEInt(file);
    std::fgetc(file);
    int cell_data = 0;
    if (!readLine(file, text) || std::sscanf(text.c_str(), "CELL_DATA %d", &cell_data) != 1 || cell_data != n_cells) return 1;
    if (!readLine(file, text) || text != "SCALARS level int 1") return 1;
    if (!readLine(file, text) || text != "LOOKUP_TABLE default") return 1;
    std::map<int, int> counts;
    for (Cell& cell : cells) { cell.level = readBEInt(file); ++counts[cell.level]; }
    std::fclose(file);

    std::printf("Archivo: %s\nCeldas: %d\nNiveles encontrados:\n", argv[1], n_cells);
    for (const auto& [level, count] : counts) std::printf("  nivel %d: %d celdas\n", level, count);
    if (counts.size() == 1) std::printf("Refinamiento: completo/uniforme (todas las celdas tienen el mismo nivel).\n");
    else std::printf("Refinamiento: por niveles/adaptativo (hay mas de un nivel).\n");

    long long violations = 0;
    if (counts.size() > 1) {
        for (size_t i = 0; i < cells.size(); ++i) for (size_t j = i + 1; j < cells.size(); ++j) {
            if (std::abs(cells[i].level - cells[j].level) > 1 && touches(cells[i], cells[j])) ++violations;
        }
    }
    if (violations == 0) std::printf("Regla 2:1: CUMPLE (sin vecinos con diferencia mayor que 1).\n");
    else std::printf("Regla 2:1: NO CUMPLE (%lld pares de vecinos en violacion).\n", violations);
    return violations == 0 ? 0 : 2;
}