# ===========================================================================
# Makefile — Octree CUDA Project
#
# Compila archivos .cu con nvcc y .cpp con g++, enlaza todo en un binario.
# La arquitectura sm_89 corresponde a Ada Lovelace (RTX 4050).
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
              $(SRC_DIR)/mesh_classify.cu $(SRC_DIR)/stats.cu
CPP_SRCS   := $(SRC_DIR)/vtk_writer.cpp

# --- Archivos objeto ---
CU_OBJS    := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
CPP_OBJS   := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
ALL_OBJS   := $(CU_OBJS) $(CPP_OBJS)

# --- Binario final ---
TARGET     := octree

# ===========================================================================
# Reglas
# ===========================================================================

.PHONY: all clean run analyze help

all: $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compilar archivos .cu
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compilar archivos .cpp (necesita include path para octree.cuh / cuda_runtime)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

# Crear directorio de build
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Crear directorio de output (usado por run)
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# ===========================================================================
# Parámetros de ejecución
#
#   LEVEL=N        Profundidad máxima del octree (1 a 20). Por defecto 3.
#   MDL=archivo    Malla de entrada. Por defecto cortex.mdl.
#   QUIET=1        Sin prints intermedios: solo el resumen y la tabla final.
#   MODE=adaptive  Refinamiento adaptativo con balance 2:1 (por defecto).
#   MODE=uniform   Refinamiento uniforme: subdivide todas las hojas.
#
# Ejemplos:
#   make run LEVEL=5
#   make run LEVEL=5 QUIET=1
#   make run LEVEL=5 MODE=uniform
#   make run LEVEL=4 MODE=uniform QUIET=1 MDL=cortex.mdl
# ===========================================================================
LEVEL ?= 3
MDL   ?= cortex.mdl
QUIET ?= 0
MODE  ?= adaptive

RUN_OPTS :=

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

run: $(TARGET) | $(OUT_DIR)
	./$(TARGET) $(LEVEL) $(MDL) $(RUN_OPTS)

analyze:
	@test -n "$(archive)" || (echo "Uso: make run analyze archive=output/archivo.vtk"; exit 1)
	$(MAKE) -C vtk_refinement_analyzer run ARCHIVE="$(abspath $(archive))"

help:
	@echo "Objetivos: all (por defecto), run, clean, help"
	@echo ""
	@echo "Variables de ejecucion:"
	@echo "  LEVEL=N        Profundidad maxima del octree (1 a 20). Default: 3"
	@echo "  MDL=archivo    Malla .mdl de entrada. Default: cortex.mdl"
	@echo "  QUIET=1        Sin prints intermedios (conserva resumen y tabla)"
	@echo "  MODE=adaptive  Balance 2:1, solo BORDER y vecinos (default)"
	@echo "  MODE=uniform   Subdivide todas las hojas en todos los niveles"
	@echo "  make run analyze archive=output/archivo.vtk  Analiza niveles y regla 2:1"
	@echo "  ARCH=sm_XX     Arquitectura CUDA destino. Default: sm_89"
	@echo ""
	@echo "Ejemplo: make run LEVEL=5 MODE=uniform QUIET=1"

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
	rm -f $(OUT_DIR)/*.vtk
