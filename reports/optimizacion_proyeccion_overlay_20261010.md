# Proyección y overlay: equivalencia y rendimiento

Configuración conservada: escala 0.5, FS4, tracking5 y volúmenes FS optimizados.
GPU RTX 4070 Laptop, video `live_20261010_030134.db3`.

## Cambios

La ruta de proyección del vivo calcula las coordenadas únicamente de los puntos
válidos, reutiliza grids de píxeles inmutables y omite las copias de la nube auxiliar
que el tracker no utiliza. Conserva el orden de las operaciones geométricas y el
z-buffer. `return_details=True` sigue siendo el valor por defecto para exportar
nubes; el vivo usa `False` y recibe `None` en el quinto valor de retorno.
Se evita además copiar puntos RGB cuando todos están delante de la cámara.

El dibujo de ejes opacos usa las mismas proyecciones, colores, orden y antialiasing
de FoundationPose, directamente sobre RGB. Elimina tres máscaras de imagen completa,
sus cálculos de norma, mezclas de transparencia y conversiones RGB/BGR. La silueta
CAD, caja y render compartido se conservan. No se modifican archivos de los modelos
de terceros. `--reference-cpu` permite comparar proyección detallada y ejes anteriores.

## Equivalencia

`projection_overlay_equivalence_v2_20261010/summary.json`: 161 fotogramas con
igualdad exacta de profundidad, validez, intrínsecos, RGB y píxeles de ejes.
Se usaron las poses FS4 guardadas, con los ejes centrados en la malla igual que
en el vivo. La función de referencia de ejes se extrajo del propio `Utils.py`.
La primera prueba, sin centrar los ejes, se conserva como historial; la validación
v2 es la utilizada aquí.

Medianas aisladas, alternando el orden A/B:

| Operación | Referencia | Optimizada |
|---|---:|---:|
| Proyección | 51.966 ms | 33.513 ms |
| Ejes | 31.094 ms | 0.122 ms |

Estos tiempos no incluyen todo el pipeline ni son FPS. El perfil previo completo
se conserva en `projection_after_volumes_20261010/projection.txt`.

## Benchmark concurrente

Fuente: `cpu_concurrent_20261010/validation_summary.json`, `summary.json` y los
cuatro JSONL con manifiestos y hashes. Se ejecutó referencia/optimizada/optimizada/
referencia sin ventana, con la misma pose inicial y frames 61 a 2465. Ambas rutas
usan los volúmenes FS optimizados. La referencia usa la proyección detallada y
los ejes anteriores; comparte la eliminación de la copia RGB cuando todos los
puntos están delante de la cámara.

| Pasada | Poses | Poses/s | LOST |
|---|---:|---:|---:|
| Referencia 1 | 318 | 3.926 | 0 |
| Optimizada 1 | 352 | 4.340 | 0 |
| Optimizada 2 | 356 | 4.387 | 0 |
| Referencia 2 | 316 | 3.897 | 0 |

| Métrica agregada | Referencia | Optimizada |
|---|---:|---:|
| Poses/s, ponderadas por duración | 3.911 | 4.364 |
| Latencia captura→publicación, mediana | 491.6 ms | 426.6 ms |
| Latencia captura→publicación, p95 | 530.7 ms | 464.7 ms |
| Proyección, mediana | 60.10 ms | 35.18 ms |
| Overlay completo, mediana | 39.51 ms | 3.10 ms |
| FS ida/vuelta, mediana | 187.83 ms | 187.18 ms |
| Tracking, mediana | 164.83 ms | 164.48 ms |

**Cadencia +11.6 %, latencia mediana -13.2 % y cero LOST en 1,342 poses.**
La latencia sigue siendo perceptible. Las etapas se solapan y sus medianas no
deben sumarse para calcular FPS. Se excluye el arranque de las estadísticas.
No se compara directamente con FPS de sesiones anteriores por variaciones de
carga; esta tabla usa únicamente las cuatro pasadas de la misma sesión.

La equivalencia numérica se verificó sobre pares fijos; el replay concurrente
selecciona frames según su velocidad y por eso las poses de cada pasada no son
comparaciones uno a uno. Cero LOST es el indicador interno, no una certificación
de exactitud absoluta. Este cambio no incluye un benchmark nuevo con cámara física
o ventana, ni pruebas en otras escenas.

## Pruebas

- Dos pruebas nuevas con subcasos de geometría vacía, dispersa, detrás de cámara,
  distorsión, ejes superpuestos y recortados; ejecutadas en ambos entornos Python.
- Tres checks geométricos existentes, seis de z-buffer, uno de render compartido
  y uno del resumen de telemetría: todos aprobados.
- Compilación sintáctica de los ocho módulos involucrados aprobada.

Los comandos para reproducir equivalencia y A/B están en `../pose/REPLAY.md`.
El modo normal activa ambos cambios sin añadir argumentos al comando de cámara.
