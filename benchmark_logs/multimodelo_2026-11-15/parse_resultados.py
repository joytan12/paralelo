#!/usr/bin/env python3
"""Convierte los logs de la campaña multimodelo en dos CSV para estadística.

  niveles.csv  -> una fila por modelo, implementación, modo y nivel completado
  resumen.csv  -> una fila por corrida: nivel máximo alcanzado, estado, tiempos y memoria

Nombres de log esperados: <modelo>_<gpu|cpu>_<adaptive|uniform>.log
(con su .time y .exit generados por run_all.sh).

Uso:  python3 parse_resultados.py [carpeta_de_logs]
Por defecto usa la carpeta donde está este script.
"""
import csv
import glob
import os
import re
import sys

CARPETA = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

RE_NOMBRE = re.compile(r'^(?P<modelo>[A-Za-z0-9]+)_(?P<impl>gpu|cpu)_(?P<modo>adaptive|uniform)'
                      r'(?:_r(?P<rep>\d+))?$')
RE_ETAPA = re.compile(r'^--- Etapa (\d+) ---')
RE_REFINA = re.compile(r'Padres refinados: (\d+) \| Hojas: (\d+) -> (\d+)')
RE_PODA = re.compile(r'Poda: (\d+) -> (\d+) hojas')
RE_MEM = re.compile(r'(?:VRAM|RAM) etapa: pico ([\d.]+) MB \| tiempo ([\d.]+) s')
RE_TABLA = re.compile(r'^\s*(\d+) \|\s+(\d+) \|\s+(\d+) \|\s+(\d+) \|')
RE_VERTS = re.compile(r'^\s*Vértices\s*: (\d+)')
RE_TRIS = re.compile(r'^\s*Triángulos: (\d+)')
RE_ELAPSED = re.compile(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): '
                        r'(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)')
RE_RSS = re.compile(r'Maximum resident set size \(kbytes\): (\d+)')


def leer_log(ruta):
    """Extrae etapas, tabla final, dimensiones del modelo y errores de un log."""
    etapas, tabla, errores = [], {}, []
    verts = tris = None
    completado = False
    cur = None
    with open(ruta, encoding='utf-8', errors='replace') as f:
        for linea in f:
            linea = linea.rstrip('\n')
            m = RE_ETAPA.match(linea)
            if m:
                cur = {'nivel': int(m.group(1))}
                etapas.append(cur)
                continue
            if cur is not None:
                m = RE_REFINA.search(linea)
                if m:
                    cur['padres'] = int(m.group(1))
                    cur['hojas_antes'] = int(m.group(2))
                    cur['hojas_refinadas'] = int(m.group(3))
                    continue
                m = RE_PODA.search(linea)
                if m:
                    cur['hojas_finales'] = int(m.group(2))
                    continue
                m = RE_MEM.search(linea)
                if m:
                    cur['pico_MB'] = float(m.group(1))
                    cur['tiempo_s'] = float(m.group(2))
                    cur = None
                    continue
            m = RE_TABLA.match(linea)
            if m:
                tabla[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
                continue
            if verts is None:
                m = RE_VERTS.match(linea)
                if m:
                    verts = int(m.group(1))
            if tris is None:
                m = RE_TRIS.match(linea)
                if m:
                    tris = int(m.group(1))
            if 'completado exitosamente' in linea:
                completado = True
            if 'CUDA error' in linea or 'malloc' in linea or 'Error' in linea:
                errores.append(linea.strip())
    return {'etapas': etapas, 'tabla': tabla, 'verts': verts, 'tris': tris,
            'completado': completado, 'errores': errores}


def leer_tiempo(ruta):
    """Devuelve (segundos de pared, RSS máximo en MB, texto completo) desde el .time."""
    if not os.path.exists(ruta):
        return None, None, ''
    with open(ruta, encoding='utf-8', errors='replace') as f:
        txt = f.read()
    wall = rss = None
    m = RE_ELAPSED.search(txt)
    if m:
        h = int(m.group(1) or 0)
        mi = int(m.group(2))
        s = float(m.group(3))
        wall = h * 3600 + mi * 60 + s
    m = RE_RSS.search(txt)
    if m:
        rss = int(m.group(1)) / 1024.0
    return wall, rss, txt


def leer_codigo(ruta):
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding='utf-8') as f:
        return int(f.read().strip())


def estado_corrida(info, codigo, txt_tiempo):
    """Describe por qué terminó la corrida, en lenguaje de límites."""
    if info['completado']:
        return 'completado'
    if codigo == 124:
        return 'límite de tiempo (timeout)'
    if codigo == 139 or 'signal 11' in txt_tiempo or 'Segmentation' in txt_tiempo:
        return 'segfault: malloc sin comprobar (memoria agotada)'
    if any('out of memory' in e for e in info['errores']):
        return 'sin VRAM (CUDA out of memory)'
    return f'error (código {codigo})'


def main():
    rutas = sorted(glob.glob(os.path.join(CARPETA, '*_*_*.log')))
    filas_niveles, filas_resumen = [], []

    for ruta in rutas:
        nombre = os.path.basename(ruta)[:-len('.log')]
        m = RE_NOMBRE.match(nombre)
        if not m:
            continue  # por ejemplo master.log
        modelo, impl, modo = m.group('modelo'), m.group('impl'), m.group('modo')
        rep = int(m.group('rep') or 1)
        impl_nombre = 'GPU (CUDA base)' if impl == 'gpu' else 'CPU multihilo (OpenMP)'

        info = leer_log(ruta)
        wall, rss_MB, txt_tiempo = leer_tiempo(ruta[:-len('.log')] + '.time')
        codigo = leer_codigo(ruta[:-len('.log')] + '.exit')

        acumulado = 0.0
        pico_max = 0.0
        nivel_max, hojas_max = 0, None
        for e in info['etapas']:
            if 'tiempo_s' not in e:
                continue  # etapa que no terminó (el fallo ocurrió dentro)
            acumulado += e['tiempo_s']
            pico_max = max(pico_max, e['pico_MB'])
            nivel_max = e['nivel']
            hojas_max = e.get('hojas_finales')
            dentro, borde = info['tabla'].get(e['nivel'], (None, None, None))[1:]
            filas_niveles.append({
                'modelo': modelo, 'implementacion': impl_nombre, 'modo': modo,
                'repeticion': rep, 'nivel': e['nivel'],
                'padres_refinados': e.get('padres'),
                'hojas_antes_refinar': e.get('hojas_antes'),
                'hojas_refinadas': e.get('hojas_refinadas'),
                'hojas_finales': e.get('hojas_finales'),
                'dentro_final': dentro, 'borde_final': borde,
                'tiempo_etapa_s': e['tiempo_s'],
                'tiempo_acumulado_s': round(acumulado, 6),
                'pico_etapa_MB': e['pico_MB'],
            })

        nivel_fallido = None
        if info['etapas'] and 'tiempo_s' not in info['etapas'][-1]:
            nivel_fallido = info['etapas'][-1]['nivel']

        error_txt = info['errores'][0] if info['errores'] else ''
        filas_resumen.append({
            'modelo': modelo, 'implementacion': impl_nombre, 'modo': modo,
            'repeticion': rep, 'vertices': info['verts'], 'triangulos': info['tris'],
            'nivel_max_completado': nivel_max,
            'hojas_nivel_max': hojas_max,
            'tiempo_etapas_s': round(acumulado, 6),
            'pico_max_etapa_MB': round(pico_max, 3),
            'tiempo_pared_s': None if wall is None else round(wall, 3),
            'rss_max_MB': None if rss_MB is None else round(rss_MB, 1),
            'codigo_salida': codigo,
            'estado': estado_corrida(info, codigo, txt_tiempo),
            'nivel_fallido': nivel_fallido,
            'primer_error': error_txt,
        })

    with open(os.path.join(CARPETA, 'niveles.csv'), 'w', newline='', encoding='utf-8') as f:
        cols = ['modelo', 'implementacion', 'modo', 'repeticion', 'nivel', 'padres_refinados',
                'hojas_antes_refinar', 'hojas_refinadas', 'hojas_finales',
                'dentro_final', 'borde_final', 'tiempo_etapa_s',
                'tiempo_acumulado_s', 'pico_etapa_MB']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(filas_niveles)

    with open(os.path.join(CARPETA, 'resumen.csv'), 'w', newline='', encoding='utf-8') as f:
        cols = ['modelo', 'implementacion', 'modo', 'repeticion', 'vertices', 'triangulos',
                'nivel_max_completado', 'hojas_nivel_max', 'tiempo_etapas_s',
                'pico_max_etapa_MB', 'tiempo_pared_s', 'rss_max_MB',
                'codigo_salida', 'estado', 'nivel_fallido', 'primer_error']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(filas_resumen)

    print(f'{len(filas_resumen)} corridas, {len(filas_niveles)} niveles -> niveles.csv, resumen.csv')
    for r in filas_resumen:
        print(f"{r['modelo']:<10} {r['implementacion']:<22} {r['modo']:<8} "
              f"nivel_max={r['nivel_max_completado']:<3} hojas={r['hojas_nivel_max']} "
              f"pared={r['tiempo_pared_s']}s rss={r['rss_max_MB']}MB -> {r['estado']}")


if __name__ == '__main__':
    main()
