# ===========================================================================
# Makefile — Octree CPU (1 hilo)
#
# Compila versión secuencial pura en C++17 sin dependencias CUDA.
# ===========================================================================

# --- Compilador ---
CXX        := g++

# --- Flags ---
CXX_FLAGS  := -O2 -std=c++17 -Wall

# --- Directorios ---
SRC_DIR    := src
BUILD_DIR  := build
OUT_DIR    := output

# --- Archivos fuente ---
SRCS       := $(SRC_DIR)/main.cpp \
              $(SRC_DIR)/octree.cpp \
              $(SRC_DIR)/mesh_classify.cpp \
              $(SRC_DIR)/vtk_writer.cpp \
              $(SRC_DIR)/stats.cpp

# --- Archivos objeto ---
OBJS       := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# --- Binario final ---
TARGET     := octree_cpu

# ===========================================================================
# Reglas
# ===========================================================================

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# ===========================================================================
# Parámetros de ejecución
#
#   LEVEL=N        Profundidad máxima del octree (1 a 20). Por defecto 3.
#   MDL=archivo    Malla de entrada. Por defecto ../cortex.mdl.
#   QUIET=1        Sin prints intermedios: solo el resumen y la tabla final.
#   MODE=adaptive  Refinamiento adaptativo con balance 2:1 (por defecto).
#   MODE=uniform   Refinamiento uniforme: subdivide todas las hojas.
#
# Ejemplos:
#   make run LEVEL=5
#   make run LEVEL=5 QUIET=1
#   make run LEVEL=5 MODE=uniform
# ===========================================================================
LEVEL ?= 3
MDL   ?= ../cortex.mdl
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

help:
	@echo "Objetivos: all (por defecto), run, clean, help"
	@echo ""
	@echo "Variables de ejecucion:"
	@echo "  LEVEL=N        Profundidad maxima del octree (1 a 20). Default: 3"
	@echo "  MDL=archivo    Malla .mdl de entrada. Default: ../cortex.mdl"
	@echo "  QUIET=1        Sin prints intermedios (conserva resumen y tabla)"
	@echo "  MODE=adaptive  Balance 2:1, solo BORDER y vecinos (default)"
	@echo "  MODE=uniform   Subdivide todas las hojas en todos los niveles"
	@echo ""
	@echo "Ejemplo: make run LEVEL=5 MODE=uniform QUIET=1"

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
	rm -f $(OUT_DIR)/*.vtk
