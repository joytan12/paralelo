#!/usr/bin/env bash
# Corre la medición de similitud (hito 4) sobre varias mallas.
# Uso: sim9.sh <nivel> <dir_salida> [opciones extra para octree_dag]
cd ~/paralelo/experimental3 || exit 1
NIVEL="$1"; OUT="$2"; shift 2
mkdir -p "$OUT"
D=../external/MixedOcTree/data
for m in cortex femur fertilityRotated cyl50 hex_sphe sein suture sc; do
  f="$D/$m.mdl"
  if [ "$m" = cortex ]; then f=../cortex.mdl; fi
  timeout 900 ./octree_dag "$NIVEL" "$f" --split-level 0 --no-vtk -q --similitud "$@" > "$OUT/$m.log" 2>&1
  echo "$m exit=$? | $(grep -E 'Validación del DAG' "$OUT/$m.log") | $(grep -E 'Costo de la medición' "$OUT/$m.log")"
done
timeout 900 ./octree_dag "$NIVEL" --esfera 0.7 --split-level 0 --no-vtk -q --similitud "$@" > "$OUT/esfera.log" 2>&1
echo "esfera exit=$? | $(grep -E 'Validación del DAG' "$OUT/esfera.log") | $(grep -E 'Costo de la medición' "$OUT/esfera.log")"
tail -1 ../benchmark_logs/multimodelo_2026-11-15/master.log
