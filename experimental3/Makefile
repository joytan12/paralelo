# ===========================================================================
# Octree CUDA experimental3 — compresión del octree final como DAG con espejos
# ===========================================================================

NVCC       := nvcc
ARCH       ?= sm_89
NVCC_FLAGS := -arch=$(ARCH) -O2 -std=c++17

SRC_DIR    := src
BUILD_DIR  := build
OUT_DIR    := output
TARGET     := octree_dag
TEST_TARGET := dag_test

CU_SRCS    := $(SRC_DIR)/main.cu $(SRC_DIR)/octree.cu $(SRC_DIR)/geometry.cu \
              $(SRC_DIR)/mesh_classify.cu $(SRC_DIR)/stats.cu $(SRC_DIR)/dag.cu \
              $(SRC_DIR)/dag_subarboles.cu $(SRC_DIR)/similitud.cu
CPP_SRCS   := $(SRC_DIR)/vtk_writer.cpp
CU_OBJS    := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
CPP_OBJS   := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
ALL_OBJS   := $(CU_OBJS) $(CPP_OBJS)

HEADERS    := $(wildcard $(SRC_DIR)/*.cuh $(SRC_DIR)/*.h $(SRC_DIR)/*.inc)

.PHONY: all run test clean help

all: $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

LEVEL  ?= 3
MDL    ?= ../cortex.mdl
QUIET  ?= 0
MODE   ?= adaptive
MIRROR ?= 1

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

ifeq ($(MIRROR),0)
RUN_OPTS += --no-mirror
else ifneq ($(MIRROR),1)
$(error MIRROR invalido: '$(MIRROR)'. Usa MIRROR=1 o MIRROR=0)
endif

# SPLIT: nivel de descomposicion del DAG por subarboles (uniform: hito 2a;
# adaptive: hito 2b, con halo de cascaras y pasadas). Si no se define, el
# binario usa 1. SPLIT=0 fuerza el camino monolitico del hito 1.
ifneq ($(origin SPLIT),undefined)
RUN_OPTS += --split-level $(SPLIT)
endif

# PASSES: pasadas maximas de cierre del balance entre subarboles (adaptive).
ifneq ($(origin PASSES),undefined)
RUN_OPTS += --passes $(PASSES)
endif

# VTK=0 no exporta el VTK (con SPLIT>0 nunca se exporta).
VTK ?= 1
ifeq ($(VTK),0)
RUN_OPTS += --no-vtk
else ifneq ($(VTK),1)
$(error VTK invalido: '$(VTK)'. Usa VTK=0 o VTK=1)
endif

# REGEN=0 omite la comparacion de cada bloque contra su regeneracion (SPLIT>0).
REGEN ?= 1
ifeq ($(REGEN),0)
RUN_OPTS += --no-regen
else ifneq ($(REGEN),1)
$(error REGEN invalido: '$(REGEN)'. Usa REGEN=0 o REGEN=1)
endif

# COMPARE=1 compara el DAG por subarboles con el monolitico (niveles <= 9).
COMPARE ?= 0
ifeq ($(COMPARE),1)
RUN_OPTS += --comparar
else ifneq ($(COMPARE),0)
$(error COMPARE invalido: '$(COMPARE)'. Usa COMPARE=0 o COMPARE=1)
endif

# PERFIL=1 acumula el tiempo por fase y por pasada (adaptive con SPLIT>0).
PERFIL ?= 0
ifeq ($(PERFIL),1)
RUN_OPTS += --perfil
else ifneq ($(PERFIL),0)
$(error PERFIL invalido: '$(PERFIL)'. Usa PERFIL=0 o PERFIL=1)
endif

# SIMILITUD=1 mide la similitud aproximada entre nodos del DAG (hito 4).
SIMILITUD ?= 0
ifeq ($(SIMILITUD),1)
RUN_OPTS += --similitud
else ifneq ($(SIMILITUD),0)
$(error SIMILITUD invalido: '$(SIMILITUD)'. Usa SIMILITUD=0 o SIMILITUD=1)
endif

# ESFERA=R reemplaza la malla por una esfera de radio R centrada en el origen
# (raiz de half_size 1, 0 < R <= 1.7).
ifneq ($(origin ESFERA),undefined)
RUN_OPTS += --esfera $(ESFERA)
endif

run: $(TARGET) | $(OUT_DIR)
	./$(TARGET) $(LEVEL) $(MDL) $(RUN_OPTS)

$(TEST_TARGET): tests/dag_test.cu $(SRC_DIR)/dag.cu $(SRC_DIR)/dag_subarboles.cu \
	$(SRC_DIR)/similitud.cu $(SRC_DIR)/octree.cu $(SRC_DIR)/geometry.cu \
	$(SRC_DIR)/stats.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -I$(SRC_DIR) -o $@ tests/dag_test.cu \
	    $(SRC_DIR)/dag.cu $(SRC_DIR)/dag_subarboles.cu $(SRC_DIR)/similitud.cu \
	    $(SRC_DIR)/octree.cu \
	    $(SRC_DIR)/geometry.cu $(SRC_DIR)/stats.cu

test: $(TEST_TARGET)
	./$(TEST_TARGET)

help:
	@echo "Objetivos: all, run, test, clean, help"
	@echo "  make run LEVEL=5 MODE=adaptive QUIET=1"
	@echo "  make run LEVEL=5 MODE=uniform MIRROR=0"
	@echo "  make run LEVEL=9 MODE=uniform SPLIT=2 QUIET=1    DAG por subarboles"
	@echo "  make run LEVEL=11 MODE=adaptive SPLIT=1 PERFIL=1 perfil de tiempos por fase"
	@echo "  make run LEVEL=9 ESFERA=0.7                       esfera de radio 0.7 en lugar de la malla"
	@echo "  make run LEVEL=9 SPLIT=0 VTK=1                    camino monolitico (exporta VTK)"
	@echo "  make test"
	@echo ""
	@echo "SPLIT=D (defecto 1; 0 = monolitico): DAG por subarboles (uniform y adaptive), un solo DAG."
	@echo "  !! Con SPLIT>0 NO se escribe VTK (las hojas completas nunca estan en VRAM)."
	@echo "  !! SPLIT=0 da el comportamiento anterior: arreglo completo de hojas, tabla de"
	@echo "     refinamiento, dos DAG (sin y con espejo) y VTK (salvo VTK=0)."
	@echo "PASSES=N: pasadas maximas de cierre del balance (adaptive, defecto 8)."
	@echo "MIRROR=1 (defecto) construye con espejo; MIRROR=0 solo sin espejo (con SPLIT=0 se"
	@echo "  construyen ambos DAG salvo MIRROR=0)."
	@echo "VTK=0 omite el VTK (solo SPLIT=0); REGEN=0 omite la regeneracion por bloque (uniform);"
	@echo "COMPARE=1 compara con el DAG monolitico (niveles <= 9 en uniform, <= 11 en adaptive)."
	@echo "PERFIL=1: tiempo por fase y pasada (adaptive, SPLIT>0). ESFERA=R: esfera en vez de malla."
	@echo "SIMILITUD=1: mide similitud aproximada entre nodos y estima un formato delta (hito 4)."
	@echo "La salida VTK (SPLIT=0) es output/octree_{pruned|uniform}_level_N.vtk"

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_TARGET)
	rm -f $(OUT_DIR)/*.vtk
