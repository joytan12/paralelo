#!/usr/bin/env bash
# Resume los logs de la medición de similitud: una línea por malla.
OUT="$1"
printf '%-17s %10s %8s %8s | %7s %7s | %7s %7s | %9s %6s %6s %6s %6s\n' \
  malla hojas unicosSE unicosCE M1d1 M1d2 M2d1 M2d2 DAG M5% cons% optC% optV%
for f in "$OUT"/*.log; do
  m=$(basename "$f" .log)
  awk -v m="$m" '
    /^=== Compresión DAG por nivel/ {sec="dag"}
    /^M1\./ {sec="m1"} /^M2\./ {sec="m2"} /^M4\./ {sec="m4"} /^Tamaños estimados/ {sec="tam"}
    sec=="dag" && /Total/ {gsub(/\|/,""); hojas=$2; se=$4; ce=$5}
    sec=="m1" && $1 ~ /^[0-9]+$/ {gsub(/\|/,""); a1+=$4; a2+=$5}
    sec=="m2" && $1 ~ /^[0-9]+$/ {gsub(/\|/,""); b1+=$3; b2+=$4}
    sec=="tam" && /DAG actual/ {dag=$(NF-1)" "$NF}
    sec=="tam" && /M5/ {match($0,/\(([0-9.]+) %\)/,x); m5=x[1]}
    sec=="tam" && /conservadora/ {match($0,/\(([0-9.]+) %\)/,x); co=x[1]}
    sec=="tam" && /formas can/ {match($0,/\(([0-9.]+) %\)/,x); oc=x[1]}
    sec=="tam" && /8 vistas/ {match($0,/\(([0-9.]+) %\)/,x); ov=x[1]}
    END {printf "%-17s %10s %8s %8s | %7d %7d | %7d %7d | %9s %6s %6s %6s %6s\n", m, hojas, se, ce, a1, a2, b1, b2, dag, m5, co, oc, ov}
  ' "$f"
done
