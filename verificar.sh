#!/usr/bin/env bash
# ===========================================================================
# verificar.sh — Comprobacion cruzada de las tres versiones.
#
# Corre CUDA, CPU 1 hilo y CPU OpenMP en los dos modos de refinamiento y
# contrasta los conteos finales. Las tres deben coincidir exactamente.
#
# Uso:  ./verificar.sh [nivel]        (nivel por defecto: 5)
# ===========================================================================
set -u
cd "$(dirname "$0")"

LEVEL="${1:-5}"

echo "Compilando las tres versiones..."
make            > /dev/null || exit 1
make -C cpu_single > /dev/null || exit 1
make -C cpu_multi  > /dev/null || exit 1
echo "Compilacion OK"
echo

printf '%-12s %-8s %s\n' "MODO" "VERSION" "RESULTADO"
printf '%-12s %-8s %s\n' "------------" "--------" "----------------------------"

for MODE in adaptive uniform; do
    if [ "$MODE" = "uniform" ]; then FLAG="--uniform"; else FLAG="--adaptive"; fi

    OUT_CUDA=$(./octree "$LEVEL" cortex.mdl -q "$FLAG" | grep 'Dentro:' | tail -1)
    OUT_SING=$(cd cpu_single && ./octree_cpu "$LEVEL" ../cortex.mdl -q "$FLAG" | grep 'Dentro:' | tail -1)
    OUT_MULT=$(cd cpu_multi && ./octree_cpu_mt "$LEVEL" ../cortex.mdl -q "$FLAG" | grep 'Dentro:' | tail -1)

    printf '%-12s %-8s %s\n' "$MODE" "cuda"   "$OUT_CUDA"
    printf '%-12s %-8s %s\n' "$MODE" "single" "$OUT_SING"
    printf '%-12s %-8s %s\n' "$MODE" "multi"  "$OUT_MULT"

    if [ "$OUT_CUDA" = "$OUT_SING" ] && [ "$OUT_CUDA" = "$OUT_MULT" ]; then
        echo "  -> las tres versiones coinciden"
    else
        echo "  -> DISCREPANCIA entre versiones"
    fi
    echo
done

echo "Archivos VTK generados para el nivel $LEVEL:"
ls -1 output/*level_"$LEVEL".vtk \
      cpu_single/output/*level_"$LEVEL".vtk \
      cpu_multi/output/*level_"$LEVEL".vtk 2>/dev/null
