#include "vtk_stream.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error en %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// --------------------------------------------------------------------------
// Secciones del archivo y su tamano por cubo.
// --------------------------------------------------------------------------
enum VtkSection {
    SEC_POINTS = 0,   // 8 vertices * 3 floats * 4 B
    SEC_CELLS  = 1,   // (1 + 8) enteros * 4 B
    SEC_TYPES  = 2,   // 1 entero
    SEC_LEVEL  = 3,   // 1 entero
    SEC_STATE  = 4,   // 1 entero
    SEC_COUNT  = 5
};

static const int kSectionBytes[SEC_COUNT] = {96, 36, 4, 4, 4};
static const char* kSectionSuffix[SEC_COUNT] = {
    ".pts.tmp", ".cells.tmp", ".types.tmp", ".level.tmp", ".state.tmp"};

struct VtkStream {
    char  path[512];
    char  tmp_path[SEC_COUNT][560];
    FILE* tmp[SEC_COUNT];

    int   chunk_cubes;
    size_t buf_bytes;

    unsigned char* d_stage[2];
    unsigned char* h_pin[2];
    cudaEvent_t    ev[2];
    cudaStream_t   stream;

    long long n_cubes;
};

// --------------------------------------------------------------------------
// Escritura big-endian en device. VTK Legacy BINARY exige big-endian
// independientemente de la arquitectura que escriba.
// --------------------------------------------------------------------------
__device__ __forceinline__ void putBE32(unsigned char* p, unsigned int v) {
    p[0] = (unsigned char)((v >> 24) & 0xffu);
    p[1] = (unsigned char)((v >> 16) & 0xffu);
    p[2] = (unsigned char)((v >> 8) & 0xffu);
    p[3] = (unsigned char)(v & 0xffu);
}

// --------------------------------------------------------------------------
// POINTS: 8 vertices por cubo, en el orden VTK_HEXAHEDRON.
// --------------------------------------------------------------------------
__global__ void secPointsKernel(const Cube* cubes, int n, unsigned char* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const Cube q = cubes[i];
    const float x0 = q.center.x - q.half_size, x1 = q.center.x + q.half_size;
    const float y0 = q.center.y - q.half_size, y1 = q.center.y + q.half_size;
    const float z0 = q.center.z - q.half_size, z1 = q.center.z + q.half_size;

    const float px[8] = {x0, x1, x1, x0, x0, x1, x1, x0};
    const float py[8] = {y0, y0, y1, y1, y0, y0, y1, y1};
    const float pz[8] = {z0, z0, z0, z0, z1, z1, z1, z1};

    unsigned char* p = out + (size_t)i * 96;
    for (int k = 0; k < 8; ++k) {
        putBE32(p + (size_t)(k * 3 + 0) * 4, __float_as_uint(px[k]));
        putBE32(p + (size_t)(k * 3 + 1) * 4, __float_as_uint(py[k]));
        putBE32(p + (size_t)(k * 3 + 2) * 4, __float_as_uint(pz[k]));
    }
}

// --------------------------------------------------------------------------
// CELLS: "8 i0 i1 ... i7". Los indices son globales, por eso index_base:
// como los subarboles se evacuan en orden, el desplazamiento ya se conoce.
// --------------------------------------------------------------------------
__global__ void secCellsKernel(int n, long long index_base,
                               unsigned char* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned char* p = out + (size_t)i * 36;
    putBE32(p, 8u);
    const long long base = (index_base + (long long)i) * 8LL;
    for (int j = 0; j < 8; ++j)
        putBE32(p + 4 + (size_t)j * 4, (unsigned int)(base + j));
}

__global__ void secTypesKernel(int n, unsigned char* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    putBE32(out + (size_t)i * 4, 12u);   // VTK_HEXAHEDRON
}

__global__ void secLevelKernel(const Cube* cubes, int n, unsigned char* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    putBE32(out + (size_t)i * 4, (unsigned int)cubes[i].level);
}

__global__ void secStateKernel(const Cube* cubes, int n, unsigned char* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    putBE32(out + (size_t)i * 4, (unsigned int)cubes[i].state);
}

// --------------------------------------------------------------------------
// Apertura y cierre.
// --------------------------------------------------------------------------
static void openTemps(VtkStream* s, const char* mode) {
    for (int k = 0; k < SEC_COUNT; ++k) {
        s->tmp[k] = fopen(s->tmp_path[k], mode);
        if (!s->tmp[k]) {
            fprintf(stderr, "Error: no se pudo abrir '%s'.\n", s->tmp_path[k]);
            exit(EXIT_FAILURE);
        }
    }
}

static void closeTemps(VtkStream* s) {
    for (int k = 0; k < SEC_COUNT; ++k) {
        if (s->tmp[k]) { fclose(s->tmp[k]); s->tmp[k] = nullptr; }
    }
}

static void removeTemps(VtkStream* s) {
    for (int k = 0; k < SEC_COUNT; ++k) remove(s->tmp_path[k]);
}

VtkStream* vtkStreamOpen(const char* out_path, int chunk_cubes) {
    if (chunk_cubes < 4096) chunk_cubes = 4096;

    VtkStream* s = (VtkStream*)calloc(1, sizeof(VtkStream));
    if (!s) { fprintf(stderr, "Error: sin memoria para VtkStream.\n"); exit(EXIT_FAILURE); }

    snprintf(s->path, sizeof(s->path), "%s", out_path);
    for (int k = 0; k < SEC_COUNT; ++k)
        snprintf(s->tmp_path[k], sizeof(s->tmp_path[k]), "%s%s",
                 out_path, kSectionSuffix[k]);

    s->chunk_cubes = chunk_cubes;
    s->buf_bytes   = (size_t)chunk_cubes * (size_t)kSectionBytes[SEC_POINTS];
    s->n_cubes     = 0;

    openTemps(s, "wb");

    CUDA_CHECK(cudaStreamCreate(&s->stream));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&s->d_stage[b], s->buf_bytes));
        CUDA_CHECK(cudaHostAlloc(&s->h_pin[b], s->buf_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaEventCreate(&s->ev[b]));
    }
    return s;
}

void vtkStreamReset(VtkStream* s) {
    if (!s) return;
    closeTemps(s);
    openTemps(s, "wb");   // "wb" trunca
    s->n_cubes = 0;
}

// --------------------------------------------------------------------------
// Evacuacion de una seccion por trozos, con anillo de dos buffers.
//
// Mientras la CPU escribe el trozo i-2 en disco, el DMA baja el i-1 y la GPU
// genera el i. Los buffers de device van en pareja para que el kernel del
// trozo i no pise el que todavia esta copiandose.
// --------------------------------------------------------------------------
static void streamSection(VtkStream* s, int section, const Cube* d_cubes,
                          int n_cubes, long long index_base) {
    if (n_cubes <= 0) return;

    FILE* out = s->tmp[section];
    const int bpc = kSectionBytes[section];
    const int threads = 256;

    const int n_chunks = (n_cubes + s->chunk_cubes - 1) / s->chunk_cubes;

    bool   pending[2]       = {false, false};
    size_t pending_bytes[2] = {0, 0};

    for (int c = 0; c < n_chunks; ++c) {
        const int b     = c & 1;
        const int start = c * s->chunk_cubes;
        const int count = (n_cubes - start < s->chunk_cubes)
                        ? (n_cubes - start) : s->chunk_cubes;
        const size_t bytes = (size_t)count * (size_t)bpc;

        // Reciclar este buffer: esperar a que su copia haya aterrizado y
        // volcarlo a disco antes de reusarlo.
        if (pending[b]) {
            CUDA_CHECK(cudaEventSynchronize(s->ev[b]));
            if (fwrite(s->h_pin[b], 1, pending_bytes[b], out) != pending_bytes[b]) {
                fprintf(stderr, "Error escribiendo la seccion VTK %d.\n", section);
                exit(EXIT_FAILURE);
            }
            pending[b] = false;
        }

        const int blocks = (count + threads - 1) / threads;
        switch (section) {
            case SEC_POINTS:
                secPointsKernel<<<blocks, threads, 0, s->stream>>>(
                    d_cubes + start, count, s->d_stage[b]);
                break;
            case SEC_CELLS:
                secCellsKernel<<<blocks, threads, 0, s->stream>>>(
                    count, index_base + start, s->d_stage[b]);
                break;
            case SEC_TYPES:
                secTypesKernel<<<blocks, threads, 0, s->stream>>>(
                    count, s->d_stage[b]);
                break;
            case SEC_LEVEL:
                secLevelKernel<<<blocks, threads, 0, s->stream>>>(
                    d_cubes + start, count, s->d_stage[b]);
                break;
            case SEC_STATE:
                secStateKernel<<<blocks, threads, 0, s->stream>>>(
                    d_cubes + start, count, s->d_stage[b]);
                break;
            default: break;
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(s->h_pin[b], s->d_stage[b], bytes,
                                   cudaMemcpyDeviceToHost, s->stream));
        CUDA_CHECK(cudaEventRecord(s->ev[b], s->stream));

        pending[b]       = true;
        pending_bytes[b] = bytes;
    }

    // Drenar en el orden en que se encolaron: primero el trozo n-2, luego n-1.
    for (int k = 0; k < 2; ++k) {
        const int b = (n_chunks + k) & 1;
        if (!pending[b]) continue;
        CUDA_CHECK(cudaEventSynchronize(s->ev[b]));
        if (fwrite(s->h_pin[b], 1, pending_bytes[b], out) != pending_bytes[b]) {
            fprintf(stderr, "Error escribiendo la seccion VTK %d.\n", section);
            exit(EXIT_FAILURE);
        }
        pending[b] = false;
    }
}

void vtkStreamAppend(VtkStream* s, const Cube* d_cubes, int n_cubes) {
    if (!s || n_cubes <= 0) return;
    const long long base = s->n_cubes;
    for (int section = 0; section < SEC_COUNT; ++section)
        streamSection(s, section, d_cubes, n_cubes, base);
    s->n_cubes += n_cubes;
}

long long vtkStreamCount(const VtkStream* s) { return s ? s->n_cubes : 0; }

// --------------------------------------------------------------------------
// Cierre: cabecera + concatenacion de las secciones.
// --------------------------------------------------------------------------
static void appendFile(FILE* dst, const char* src_path, unsigned char* buf,
                       size_t buf_size) {
    FILE* src = fopen(src_path, "rb");
    if (!src) {
        fprintf(stderr, "Error: no se pudo releer '%s'.\n", src_path);
        exit(EXIT_FAILURE);
    }
    size_t got;
    while ((got = fread(buf, 1, buf_size, src)) > 0) {
        if (fwrite(buf, 1, got, dst) != got) {
            fprintf(stderr, "Error concatenando '%s'.\n", src_path);
            exit(EXIT_FAILURE);
        }
    }
    fclose(src);
}

static void releaseDevice(VtkStream* s) {
    for (int b = 0; b < 2; ++b) {
        if (s->d_stage[b]) { cudaFree(s->d_stage[b]); s->d_stage[b] = nullptr; }
        if (s->h_pin[b])   { cudaFreeHost(s->h_pin[b]); s->h_pin[b] = nullptr; }
        if (s->ev[b])      { cudaEventDestroy(s->ev[b]); s->ev[b] = nullptr; }
    }
    if (s->stream) { cudaStreamDestroy(s->stream); s->stream = nullptr; }
}

void vtkStreamFinish(VtkStream* s) {
    if (!s) return;
    closeTemps(s);

    const long long nc = s->n_cubes;
    const long long np = nc * 8LL;

    FILE* fp = fopen(s->path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: no se pudo abrir '%s' para escritura.\n", s->path);
        exit(EXIT_FAILURE);
    }

    // Se reusan los buffers pinned como buffer de copia: ya estan reservados.
    unsigned char* buf = s->h_pin[0];
    const size_t   cap = s->buf_bytes;

    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Octree refinement level output\n");
    fprintf(fp, "BINARY\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");

    fprintf(fp, "POINTS %lld float\n", np);
    appendFile(fp, s->tmp_path[SEC_POINTS], buf, cap);
    fputc('\n', fp);

    fprintf(fp, "CELLS %lld %lld\n", nc, nc * 9LL);
    appendFile(fp, s->tmp_path[SEC_CELLS], buf, cap);
    fputc('\n', fp);

    fprintf(fp, "CELL_TYPES %lld\n", nc);
    appendFile(fp, s->tmp_path[SEC_TYPES], buf, cap);
    fputc('\n', fp);

    fprintf(fp, "CELL_DATA %lld\n", nc);

    fprintf(fp, "SCALARS level int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    appendFile(fp, s->tmp_path[SEC_LEVEL], buf, cap);
    fputc('\n', fp);

    fprintf(fp, "SCALARS state int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    appendFile(fp, s->tmp_path[SEC_STATE], buf, cap);
    fputc('\n', fp);

    fclose(fp);

    removeTemps(s);
    releaseDevice(s);
    printf("VTK binario exportado: %s  (%lld cubos, %lld puntos)\n",
           s->path, nc, np);
    free(s);
}

void vtkStreamDiscard(VtkStream* s) {
    if (!s) return;
    closeTemps(s);
    removeTemps(s);
    releaseDevice(s);
    free(s);
}

long long vtkEstimatedBytes(long long n_cubes) {
    long long per_cube = 0;
    for (int k = 0; k < SEC_COUNT; ++k) per_cube += kSectionBytes[k];
    return n_cubes * per_cube + 512;
}

long long vtkMaxCubes() {
    // Los indices de punto se escriben como int32 y hay 8 puntos por cubo.
    return 2147483647LL / 8LL;
}
