#ifndef MESH_H
#define MESH_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <cmath>

// --------------------------------------------------------------------------
// Estructura que representa una malla triangular cargada desde un .mdl.
// --------------------------------------------------------------------------
struct Mesh {
    float3* vertices;   // arreglo de vértices (host)
    int3*   triangles;  // arreglo de triángulos: cada int3 = (i0, i1, i2) (host)
    int     n_verts;
    int     n_tris;

    // Bounding box
    float3  bbox_min;
    float3  bbox_max;
    float3  center;
    float   half_size;  // mitad del lado del cubo envolvente cúbico (mayor de los 3 ejes)
};

// --------------------------------------------------------------------------
// Libera la memoria del host de la malla.
// --------------------------------------------------------------------------
inline void freeMesh(Mesh& m) {
    free(m.vertices);
    free(m.triangles);
    m.vertices  = nullptr;
    m.triangles = nullptr;
    m.n_verts   = 0;
    m.n_tris    = 0;
}

// --------------------------------------------------------------------------
// Parsea un archivo .mdl con el formato:
//
//   [Vertices, ARRAY1<POINT3D>]
//   N
//
//    +X +Y +Z
//   ...
//
//   [Triangles, ARRAY1<STRING>]
//   M
//
//   i0 i1 i2 0 0 0
//   ...
//
// Devuelve true si la carga fue exitosa.
// --------------------------------------------------------------------------
inline bool loadMDL(const char* path, Mesh& mesh) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[MDL] Error: no se puede abrir '%s'\n", path);
        return false;
    }

    mesh.vertices  = nullptr;
    mesh.triangles = nullptr;
    mesh.n_verts   = 0;
    mesh.n_tris    = 0;

    char line[256];
    bool in_verts = false, in_tris = false;
    int  v_read   = 0,     t_read  = 0;

    while (fgets(line, sizeof(line), fp)) {
        // Eliminar salto de línea
        char* nl = strchr(line, '\n');
        if (nl) *nl = '\0';

        // --- Detectar sección de vértices ---
        if (strstr(line, "[Vertices,")) {
            in_verts = true;
            in_tris  = false;
            // Siguiente línea: cantidad de vértices
            if (fgets(line, sizeof(line), fp)) {
                mesh.n_verts = atoi(line);
                mesh.vertices = (float3*)malloc(mesh.n_verts * sizeof(float3));
                if (!mesh.vertices) {
                    fprintf(stderr, "[MDL] Error: malloc vértices falló\n");
                    fclose(fp);
                    return false;
                }
            }
            continue;
        }

        // --- Detectar sección de triángulos ---
        if (strstr(line, "[Triangles,")) {
            in_verts = false;
            in_tris  = true;
            // Siguiente línea: cantidad de triángulos
            if (fgets(line, sizeof(line), fp)) {
                mesh.n_tris = atoi(line);
                mesh.triangles = (int3*)malloc(mesh.n_tris * sizeof(int3));
                if (!mesh.triangles) {
                    fprintf(stderr, "[MDL] Error: malloc triángulos falló\n");
                    fclose(fp);
                    return false;
                }
            }
            continue;
        }

        // Ignorar líneas vacías
        bool empty = true;
        for (int k = 0; line[k]; k++) {
            if (line[k] != ' ' && line[k] != '\t' && line[k] != '\r') {
                empty = false;
                break;
            }
        }
        if (empty) continue;

        // --- Leer vértices ---
        if (in_verts && v_read < mesh.n_verts) {
            float x, y, z;
            if (sscanf(line, "%f %f %f", &x, &y, &z) == 3) {
                mesh.vertices[v_read++] = make_float3(x, y, z);
            }
            continue;
        }

        // --- Leer triángulos ---
        if (in_tris && t_read < mesh.n_tris) {
            int i0, i1, i2, dummy;
            if (sscanf(line, "%d %d %d %d %d %d", &i0, &i1, &i2, &dummy, &dummy, &dummy) >= 3) {
                mesh.triangles[t_read++] = make_int3(i0, i1, i2);
            }
            continue;
        }
    }
    fclose(fp);

    if (v_read != mesh.n_verts || t_read != mesh.n_tris) {
        fprintf(stderr, "[MDL] Advertencia: esperados %d vértices / %d triángulos, "
                        "leídos %d / %d\n",
                mesh.n_verts, mesh.n_tris, v_read, t_read);
        mesh.n_verts = v_read;
        mesh.n_tris  = t_read;
    }

    // --- Calcular bounding box ---
    mesh.bbox_min = make_float3( FLT_MAX,  FLT_MAX,  FLT_MAX);
    mesh.bbox_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < mesh.n_verts; i++) {
        float3 v = mesh.vertices[i];
        if (v.x < mesh.bbox_min.x) mesh.bbox_min.x = v.x;
        if (v.y < mesh.bbox_min.y) mesh.bbox_min.y = v.y;
        if (v.z < mesh.bbox_min.z) mesh.bbox_min.z = v.z;
        if (v.x > mesh.bbox_max.x) mesh.bbox_max.x = v.x;
        if (v.y > mesh.bbox_max.y) mesh.bbox_max.y = v.y;
        if (v.z > mesh.bbox_max.z) mesh.bbox_max.z = v.z;
    }

    // Centro de la bbox
    mesh.center.x = (mesh.bbox_min.x + mesh.bbox_max.x) * 0.5f;
    mesh.center.y = (mesh.bbox_min.y + mesh.bbox_max.y) * 0.5f;
    mesh.center.z = (mesh.bbox_min.z + mesh.bbox_max.z) * 0.5f;

    // Half-size: el mayor de los 3 semiejes + 5% de margen
    float hx = (mesh.bbox_max.x - mesh.bbox_min.x) * 0.5f;
    float hy = (mesh.bbox_max.y - mesh.bbox_min.y) * 0.5f;
    float hz = (mesh.bbox_max.z - mesh.bbox_min.z) * 0.5f;
    float max_h = hx > hy ? hx : hy;
    if (hz > max_h) max_h = hz;
    mesh.half_size = max_h * 1.05f;  // 5% de margen

    printf("[MDL] Cargado '%s'\n", path);
    printf("      Vértices : %d\n", mesh.n_verts);
    printf("      Triángulos: %d\n", mesh.n_tris);
    printf("      BBox min  : (%.2f, %.2f, %.2f)\n",
           mesh.bbox_min.x, mesh.bbox_min.y, mesh.bbox_min.z);
    printf("      BBox max  : (%.2f, %.2f, %.2f)\n",
           mesh.bbox_max.x, mesh.bbox_max.y, mesh.bbox_max.z);
    printf("      Centro    : (%.2f, %.2f, %.2f)\n",
           mesh.center.x, mesh.center.y, mesh.center.z);
    printf("      Half-size : %.2f\n\n", mesh.half_size);

    return true;
}

#endif // MESH_H
