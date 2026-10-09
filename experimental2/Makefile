# ===========================================================================
# Octree CUDA experimental2 — unión final de regiones del mismo nivel
# ===========================================================================

NVCC       := nvcc
ARCH       ?= sm_89
NVCC_FLAGS := -arch=$(ARCH) -O2 -std=c++17

SRC_DIR    := src
BUILD_DIR  := build
OUT_DIR    := output
TARGET     := octree_merged
TEST_TARGET := merge_test

CU_SRCS    := $(SRC_DIR)/main.cu $(SRC_DIR)/octree.cu $(SRC_DIR)/geometry.cu \
              $(SRC_DIR)/mesh_classify.cu $(SRC_DIR)/stats.cu $(SRC_DIR)/merge.cu
CPP_SRCS   := $(SRC_DIR)/vtk_merged_writer.cpp
CU_OBJS    := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
CPP_OBJS   := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
ALL_OBJS   := $(CU_OBJS) $(CPP_OBJS)

.PHONY: all run test clean help

all: $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

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

$(TEST_TARGET): tests/merge_test.cu $(SRC_DIR)/merge.cu $(SRC_DIR)/merge.cuh \
	$(SRC_DIR)/merged_face.cuh $(SRC_DIR)/octree.cuh
	$(NVCC) $(NVCC_FLAGS) -I$(SRC_DIR) -o $@ tests/merge_test.cu $(SRC_DIR)/merge.cu

test: $(TEST_TARGET)
	./$(TEST_TARGET)

help:
	@echo "Objetivos: all, run, test, clean, help"
	@echo "  make run LEVEL=5 MODE=adaptive QUIET=1"
	@echo "  make run LEVEL=5 MODE=uniform"
	@echo "  make test"
	@echo "La salida es output/octree_merged_<modo>_level_N.vtk"

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_TARGET)
	rm -f $(OUT_DIR)/*.vtk
