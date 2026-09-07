CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pedantic
TARGET := vtk_refinement_analyzer

.PHONY: all clean run

all: $(TARGET)

$(TARGET): analyze.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	@test -n "$(ARCHIVE)" || (echo "Uso: make run ARCHIVE=../output/archivo.vtk"; exit 1)
	./$(TARGET) "$(ARCHIVE)"

clean:
	rm -f $(TARGET)