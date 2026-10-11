# Construcción de volúmenes de FoundationStereo

Configuración conservada: escala 0.5, FS4 y tracking5, RTX 4070 Laptop de 8 GB.

## Cambio

`scripts/fs_volumes.py` normaliza una vez cada mapa de características para la
correlación por grupos. Antes se repetía la normalización para cada disparidad.
La multiplicación y suma por grupo conservan el orden de reducción original.
La concatenación construye los desplazamientos del mapa derecho con índices
vectorizados y conserva el mapa izquierdo completo, incluso donde x < disparidad,
tal como hace la implementación original.

`scripts/fs_model.py` instala estas funciones al cargar el modelo. No se modifican
pesos ni archivos de FoundationStereo en `third_party`. `--reference-volumes`
permite ejecutar el camino original desde `pose/live_pose.py`.

## Equivalencia y tiempos aislados

Fuente: `volume_equivalence_20261010/summary.json` y `frames.json`.
La prueba ejecutó ambos caminos, alternando el orden, sobre los mismos 161 pares
IR del replay fijo. Las 161 disparidades coinciden exactamente (diferencia máxima
0 píxeles). Los volúmenes también coinciden exactamente en las características
reales capturadas del primer par. Esta equivalencia no es una medición de exactitud
absoluta frente a una referencia externa.

| Mediana | Original | Optimizado |
|---|---:|---:|
| Volumen de correlación | 18.085 ms | 4.209 ms |
| Volumen de concatenación | 5.244 ms | 0.369 ms |
| Inferencia FS completa | 160 ms | 135 ms |

Los volúmenes se midieron con eventos CUDA, 3 calentamientos y 20 repeticiones
alternadas; la inferencia completa incluye los 161 pares con 3 calentamientos por
camino. Son mediciones separadas: no se suman a los tiempos concurrentes del vivo.

## Resultado concurrente

Fuente: `volume_concurrent_20261010/validation_summary.json`, `summary.json`,
los cuatro JSONL y sus manifiestos. Todas las pasadas cubrieron los frames
61 a 2465 del video original.

| Pasada | Poses | Poses/s | LOST |
|---|---:|---:|---:|
| Original 1 | 275 | 3.394 | 0 |
| Optimizada 1 | 302 | 3.725 | 0 |
| Optimizada 2 | 301 | 3.707 | 0 |
| Original 2 | 292 | 3.604 | 0 |

| Métrica agregada | Original | Optimizado |
|---|---:|---:|
| Poses/s, ponderadas por duración | 3.499 | 3.716 |
| Latencia captura→publicación, mediana | 514.0 ms | 506.0 ms |
| Latencia captura→publicación, p95 | 570.7 ms | 556.6 ms |
| FS ida/vuelta, mediana | 206.6 ms | 191.8 ms |

**Mejora de cadencia: 6.2 %. Cero LOST en 1,170 resultados.** La cadencia es
menor que la mejora aislada de FS porque proyección, tracking y dibujo conservan
su costo y comparten recursos. Las etapas se solapan: no sumar sus medianas.
La variación entre referencias (3.394 y 3.604) muestra por qué se compara dentro
de la misma sesión y se reportan las cuatro pasadas. Estos datos no sustituyen
un benchmark nuevo con cámara/ventana, ni garantizan exactitud absoluta.

## Verificación

- Disparidad idéntica en los 161 pares registrados; no se redujo calidad numérica
  en esta secuencia.
- 2 pruebas de volúmenes con subcasos CPU/CUDA, float32/float16/bfloat16,
  entradas no contiguas, ceros, disparidades vacías y mayores al ancho; tamaños
  de características correspondientes a escalas 0.4, 0.5 y 1.0 en CUDA.
- 6 pruebas del transporte FS y 1 de agregación de telemetría aprobadas.
- Compilación sintáctica de los módulos modificados aprobada.

## Reproducción

Los comandos completos y las variantes están en `../pose/REPLAY.md`, sección
«Volúmenes estéreo equivalentes». La prueba concurrente usa el video original,
una pose inicial común, ventana desactivada y cuatro pasadas en orden
original/optimizado/optimizado/original. Su duración medida excluye el arranque.
Los hashes de código y los argumentos se guardan con la telemetría.

El comando de cámara conserva sus argumentos:

```powershell
uv run --project pose python pose/live_pose.py --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --scale 0.5 --valid-iters 4 --track-iters 5 --mask-text 'computer mouse'
```
