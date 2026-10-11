# FoundationPose: perfil, optimización y validación de tracking

Escala 0.5, FS4, tracking5; se conservan las optimizaciones de volúmenes FS,
proyección y overlay. Hardware: RTX 4070 Laptop, 8 GB.

## 1. Perfil interno

`tracking_profile_20261010/summary.json` y `tracking.txt` registran diez llamadas
con cinco iteraciones, después de tres calentamientos, sobre un bundle FS4 real.
Se usaron cProfile y eventos CUDA por etapa. Los eventos incluyen huecos de envío
al stream y las etapas anidadas se solapan; no son tiempos exclusivos de kernels.
El perfil instrumentado dio una mediana total de 100.8 ms, que no debe compararse
directamente con el tiempo de tracking concurrente.

| Etapa, acumulada por llamada | Mediana |
|---|---:|
| Preparación completa de recortes | 58.88 ms |
| Render dentro de los recortes | 16.49 ms |
| Warp de RGB y XYZ | 16.37 ms |
| Cálculo de ventanas | 4.74 ms |
| Transformación de los datos del recorte | 4.79 ms |
| Forward de la red, cinco iteraciones | 27.65 ms |
| Vaciado de caché CUDA | 0.84 ms |

El mayor costo estaba en preparar los recortes. El vaciado de caché, por sí solo,
no era el principal costo; además obliga a volver a reservar buffers posteriormente.

## 2. Cambios

- Reutilización de la misma rejilla de coordenadas para muestrear RGB y XYZ dentro
  de cada recorte. Conserva las operaciones de normalización/inversión de Kornia,
  el muestreo bilinear/nearest y `align_corners=False`. La rejilla se descarta al
  salir del recorte; no se reutiliza entre poses.
- La pose permanece en GPU al entrar al predictor; se elimina su copia GPU→CPU→GPU.
- Se conserva la caché del allocator CUDA al terminar el tracking.

`pose/tracking_fast.py` crea una copia local del predictor instalado, valida las
dos llamadas `empty_cache` esperadas y elimina únicamente esas llamadas. La copia
usa un constructor de recortes con globals propios. Los módulos de FoundationPose,
Kornia y `torch.cuda.empty_cache` originales permanecen intactos. No se mantienen
dos copias manuales del algoritmo de refinamiento.

El registro global continúa usando el predictor original. El tracking con debug
visual también conserva el camino original. `--reference-tracking` y el argumento
`reference=True` en `track_pose` permiten ejecutar la referencia. No se reducen
iteraciones, tamaño de recortes, resolución de entrada ni precisión de la red.

## 3. Equivalencia en secuencias fijas

`tracking_equivalence_20261010/summary.json`: cuatro secuencias independientes
en orden referencia/optimizada/optimizada/referencia, con los mismos 160 frames
de tracking FS4 y la misma pose inicial. Cada secuencia alimenta el siguiente paso
con su propia estimación, sin sustituirla por una pose de referencia.

Las matrices 4×4 coincidieron **exactamente en todos los frames y repeticiones**:
diferencia máxima entre elementos 0.0. Hubo cero LOST en los 640 pasos evaluados.
La memoria máxima reservada por el allocator de FoundationPose fue 188 MiB en
referencia y 212 MiB en la candidata; no es la memoria total de GPU del sistema.

Los tiempos de estas secuencias aisladas tuvieron una deriva fuerte (referencia
142.4→76.2 ms; candidata 116.9→86.1 ms de mediana). Por ello no se atribuye una
ganancia porcentual a esos tiempos; la evaluación de velocidad usa una medición
alternada sin instrumentación y el A/B concurrente completo.

## Medición alternada sin instrumentación

`tracking_paired_20261010/paired.json`: veinte muestras por variante sobre el mismo
frame y estado inicial, alternando el orden y calentando cada variante antes de
medir. Medianas: **82.46 ms referencia y 71.37 ms optimizada (-13.4 %)**. P95:
109.71 y 91.98 ms. Esta medición confirma la reducción del trabajo de tracking,
pero no representa FPS con FS ejecutándose a la vez.

## A/B concurrente completo

`tracking_concurrent_20261010/validation_summary.json`, los JSONL y sus manifiestos:
cuatro pasadas sin ventana, misma pose inicial, frames 61 a 2465, FS4 y tracking5.

| Pasada | Poses | Poses/s | LOST |
|---|---:|---:|---:|
| Referencia 1 | 355 | 4.381 | 0 |
| Optimizada 1 | 349 | 4.304 | 0 |
| Optimizada 2 | 351 | 4.327 | 0 |
| Referencia 2 | 342 | 4.228 | 0 |

| Métrica agregada | Referencia | Optimizada |
|---|---:|---:|
| Poses/s ponderadas por duración | 4.305 | 4.315 |
| Tracking, mediana | 166.64 ms | 148.58 ms |
| Latencia captura→publicación, mediana | 430.04 ms | 410.97 ms |
| Latencia captura→publicación, p95 | 477.86 ms | 454.28 ms |
| FS ida/vuelta, mediana | 193.69 ms | 192.74 ms |
| Proyección, mediana | 31.98 ms | 32.88 ms |

**Tracking -10.8 %, latencia mediana -4.4 %, FPS prácticamente iguales.**
La diferencia de cadencia (+0.25 %) es pequeña frente a la variación entre
pasadas; no se presenta como una mejora demostrada de FPS. El productor de frames
(FS, profundidad y proyección) sigue marcando el intervalo de llegada de bundles;
reducir tracking acorta la latencia sin aumentar apreciablemente esa cadencia.

Hubo **cero LOST en 1,397 poses** concurrentes. La memoria reservada del allocator
FP se mantuvo en 292 MiB durante ambas pasadas optimizadas, frente a 184–186 MiB
en referencia. Es una reserva reutilizable y acotada en estas pasadas, no una
medición de toda la VRAM del sistema. Se conserva el cambio activo por la reducción
de latencia y la equivalencia exacta, manteniendo disponible la referencia.

## Comprobaciones y límites

- Cuatro pruebas del adaptador: igualdad de warps CPU/CUDA, reutilización efectiva
  de la rejilla, invalidación al cambiar geometría, limpieza entre recortes,
  conservación del predictor original y rechazo de estructura inesperada.
- Una prueba del resumen de telemetría; compilación de ocho módulos: aprobadas.
- 640 pasos offline con poses idénticas, además del A/B concurrente.

LOST es una métrica interna, no exactitud frente a ground truth. Las pasadas
concurrentes no procesan exactamente los mismos frames intermedios: la equivalencia
uno a uno se evaluó separadamente con los bundles fijos. Esta validación usa una
grabación; no incluye un nuevo ensayo de cámara física, ventana o escenas distintas.
Las cinco iteraciones se conservan y el comando normal de cámara no cambia.

Los comandos de reproducción se encuentran en `../pose/REPLAY.md`.
