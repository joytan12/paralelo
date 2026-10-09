# Oportunidades de mejora para la versión GPU

Revisión de la versión CUDA de la raíz (`src/`) y sus variantes experimentales.

| Prioridad | Oportunidad | Motivo |
|---|---|---|
| **Muy alta** | **Acelerar la clasificación con un BVH de triángulos.** | Hoy cada cubo recorre los triángulos para buscar intersecciones y, si no encuentra ninguna, vuelve a recorrerlos para el ray casting. Un BVH construido una vez permitiría revisar solo los triángulos cercanos. Es el mayor candidato para reducir el tiempo de cálculo en [`src/mesh_classify.cu`](src/mesh_classify.cu#L215). |
| **Alta** | **Heredar el estado `INSIDE` de los hijos.** | Al generar hijos, todos quedan sin clasificar y el programa vuelve a hacer la clasificación geométrica. Si el padre ya era `INSIDE`, sus hijos también lo son bajo la definición actual del clasificador; se podría clasificarlos directamente y reservar el trabajo geométrico para hijos de padres `BORDER`. Los hijos se crean en [`src/octree.cu`](src/octree.cu#L55) y luego se clasifican en [`src/main.cu`](src/main.cu#L244). |
| **Alta** | **Reducir el costo del balance 2:1.** | En cada etapa se ordenan las claves de todas las hojas, se buscan vecinos con búsqueda binaria y se ejecutan hasta `max_level` rondas con sincronización. Conviene medir ese tramo y evaluar terminar cuando ya no haya cambios, o usar una estructura de vecinos que reduzca búsquedas y ordenamientos. Está en [`src/octree.cu`](src/octree.cu#L241). |
| **Alta para tarjetas con poca VRAM** | **Evitar reservar un segundo arreglo completo durante la poda.** | La versión base reserva espacio para tantos cubos como recibe, aunque muchos se descarten, y compacta después. La variante `experimental` ya explora procesamiento por subárboles y reducción de memoria. En las mediciones del repositorio, al nivel 11 bajó el pico de 3,22 GB a 464,72 MB, pero tardó aproximadamente el doble. [`Poda base`](src/octree.cu#L314) · [comparación medida](experimental/bench/BENCHMARK.md#L70). |
| **Media** | **Reutilizar memoria y quitar sincronizaciones innecesarias.** | Hay varias reservas y liberaciones por etapa, además de sincronizaciones explícitas tras kernels. Reutilizar buffers y sincronizar solo cuando el host necesite un resultado puede reducir pausas y costo de gestión. Primero conviene medir el efecto: el código registra tiempos por etapa, pero no separa cada kernel. [`Sincronización de clasificación`](src/mesh_classify.cu#L245) · [medición por etapa](src/main.cu#L214). |
| **Media si el cuello es la exportación** | **Transmitir el VTK por bloques.** | La versión base copia todos los cubos a RAM y los escribe campo por campo. `experimental` ya tiene una ruta con bloques y buffers pinned que se pueden solapar con la escritura. Esto ayuda al uso de RAM y al tiempo de salida, aunque no acelera el refinamiento. [`Copia base al host`](src/main.cu#L293) · [`Escritura por bloques experimental`](experimental/src/vtk_stream.cu#L193). |

## Orden sugerido

1. Perfilar clasificación, balance y poda por separado.
2. Implementar el BVH y la herencia de `INSIDE`.
3. Si el límite es la VRAM, estudiar la poda en sitio y el procesamiento por subárboles.
4. Revisar la exportación si se trabaja con resultados grandes; consume memoria de host y el formato VTK Legacy tiene límites de índices.

La variante experimental documenta que `SPLIT>1` aún requiere más validación del balance con los valores predeterminados.
