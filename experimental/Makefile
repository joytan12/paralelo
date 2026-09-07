# ===========================================================================
# Makefile — Octree CUDA (experimental: descomposicion por subarboles)
#
# Mismo esquema que el proyecto base. Cambian las fuentes: el escritor VTK
# de host (vtk_writer.cpp) se reemplaza por el escritor por streaming desde
# VRAM (vtk_stream.cu).
# ===========================================================================

# --- Compiladores ---
NVCC       := nvcc
CXX        := g++

# --- Flags ---
ARCH       ?= sm_89
NVCC_FLAGS := -arch=$(ARCH) -O2
CXX_FLAGS  := -O2 -std=c++17

# --- Directorios ---
SRC_DIR    := src
BUILD_DIR  := build
OUT_DIR    := output

# --- Archivos fuente ---
CU_SRCS    := $(SRC_DIR)/main.cu $(SRC_DIR)/octree.cu $(SRC_DIR)/geometry.cu \
              $(SRC_DIR)/mesh_classify.cu $(SRC_DIR)/stats.cu \
              $(SRC_DIR)/vtk_stream.cu

# --- Archivos objeto ---
CU_OBJS    := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
ALL_OBJS   := $(CU_OBJS)

# --- Binario final ---
TARGET     := octree

# ===========================================================================
# Reglas
# ===========================================================================

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# ===========================================================================
# Parametros de ejecucion
#
#   LEVEL=N        Profundidad maxima del octree (1 a 20). Por defecto 3.
#   MDL=archivo    Malla de entrada. Por defecto cortex.mdl.
#   QUIET=1        Sin prints intermedios.
#   MODE=adaptive  Refinamiento adaptativo con balance 2:1 (por defecto).
#   MODE=uniform   Refinamiento uniforme.
#   SPLIT=N        Nivel de descomposicion: 8^N subarboles. Por defecto 1.
#   PASSES=N       Pasadas externas para cerrar el balance 2:1. Por defecto 3.
#   VTK=0          No escribir el archivo VTK.
#
# Ejemplos:
#   make run LEVEL=8
#   make run LEVEL=12 SPLIT=1 QUIET=1
#   make run LEVEL=13 SPLIT=2 VTK=0
# ===========================================================================
LEVEL  ?= 3
MDL    ?= cortex.mdl
QUIET  ?= 0
MODE   ?= adaptive
SPLIT  ?= 1
PASSES ?= 3
VTK    ?= 1

RUN_OPTS := --split-level $(SPLIT) --passes $(PASSES)

ifeq ($(QUIET),1)
RUN_OPTS += --quiet
else ifneq ($(QUIET),0)
$(error QUIET invalido: '$(QUIET)'. Usa QUIET=0 o QUIET=1)
endif

ifeq ($(MODE),adaptive)
RUN_OPTS += --adaptive
else ifeq ($(MODE),uniform)
RUN_OPTS += --uniform
else
$(error MODE invalido: '$(MODE)'. Usa MODE=adaptive o MODE=uniform)
endif

ifeq ($(VTK),0)
RUN_OPTS += --no-vtk
else ifneq ($(VTK),1)
$(error VTK invalido: '$(VTK)'. Usa VTK=0 o VTK=1)
endif

run: $(TARGET) | $(OUT_DIR)
	./$(TARGET) $(LEVEL) $(MDL) $(RUN_OPTS)

help:
	@echo "Objetivos: all (por defecto), run, clean, help"
	@echo ""
	@echo "Variables de ejecucion:"
	@echo "  LEVEL=N        Profundidad maxima del octree (1 a 20). Default: 3"
	@echo "  MDL=archivo    Malla .mdl de entrada. Default: cortex.mdl"
	@echo "  QUIET=1        Sin prints intermedios"
	@echo "  MODE=adaptive  Balance 2:1, solo BORDER y vecinos (default)"
	@echo "  MODE=uniform   Subdivide todas las hojas en todos los niveles"
	@echo "  SPLIT=N        Nivel de descomposicion: 8^N subarboles. Default: 1"
	@echo "  PASSES=N       Pasadas externas de cierre del balance. Default: 3"
	@echo "  VTK=0          No escribe el archivo VTK"
	@echo "  ARCH=sm_XX     Arquitectura CUDA destino. Default: sm_89"
	@echo ""
	@echo "Ejemplo: make run LEVEL=12 SPLIT=1 QUIET=1"

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
	rm -f $(OUT_DIR)/*.vtk $(OUT_DIR)/*.tmp
