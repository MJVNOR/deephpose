# Z-buffer sin doble ordenamiento — 2026-10-10

## Resultado del pipeline completo

**14.5% más actualizaciones de pose**, de 3.49 a 4.00 poses/s, con FS4,
tracking5 y escala0.5 idénticos en ambas variantes. Cero LOST en los
1,215 resultados de cuatro reproducciones completas.

| Z-buffer | Pasadas | Resultados | Poses/s agrupadas | Latencia p50 | Latencia p95 | LOST |
|---|---:|---:|---:|---:|---:|---:|
| Ordenamiento anterior | 2 | 567 | 3.494 | 507 ms | 574 ms | 0 |
| Reducciones indexadas | 2 | 648 | 4.000 | 469 ms | 525 ms | 0 |

Orden ABBA: anterior 3.401, nuevo 3.997, nuevo 4.003, anterior 3.587
poses/s. Cada pasada cubrió IR 61–2465. Se restauró la misma pose inicial,
se calentaron modelos con fuente pausada y se reprodujo a velocidad real.
Headless, con overlay por pose, sin escritura de muestras ni otras pruebas
GPU simultáneas. SAM/registro y dibujo de ventana quedan fuera de esta
comparación. Los resultados finales son 4.00 poses/s, no 4 FPS de video.

Throughput agrupado por tiempo entre publicaciones; se excluye la primera
latencia. Latencia desde recepción del frame a publicación, sin incluir
exposición física ni presentación en pantalla. La variación entre las dos
referencias muestra por qué se compara dentro de la misma sesión.

En concurrencia, la proyección bajó de 77.36 a 47.79 ms de mediana. Tracking
se mantuvo en 160.2 ms y el roundtrip FS pasó de 199.5 a 195.4 ms. Esos
tiempos reflejan solapamiento y contención; no se suman para obtener FPS.
FS sigue siendo el mayor costo del hilo de profundidad.

Evidencia: `zbuffer_concurrent_20261010/validation_summary.json`, cuatro
JSONL, manifiestos de código y logs. La proyección idéntica sobre entradas
fijas no implica poses idénticas entre pasadas a tiempo real: al aumentar
throughput también cambia qué frames procesa el tracker.

## Cambio

`scripts/project_to_rgb.py` reemplaza la ordenación por profundidad y el
ordenamiento implícito de `unique` por dos reducciones `np.minimum.at`:
primero profundidad mínima por píxel; después índice mínimo entre los
puntos que empatan en esa profundidad. Al final recupera el valor del punto
ganador, conservando incluso el signo del cero. Píxeles vacíos: `inf/-1`.

La complejidad del camino de datos finitos es lineal en puntos y píxeles.
La ruta de ordenamiento se conserva para valores no finitos y para A/B
(`--sorted-zbuffer`). No se cambian escala, iteraciones, calibración,
rectificación, transformaciones, selección de puntos ni render.

## Equivalencia y medición aislada

Se recalculó profundidad FS4 a escala 0.5 sobre los 161 pares IR congelados
del video original. Cada profundidad se proyectó por ambos caminos,
alternando su orden por frame y calentando ambas rutas al inicio.

**161/161 frames idénticos valor a valor** en profundidad, máscara válida,
intrínsecos, RGB rectificado, coordenadas proyectadas, profundidad en cámara
RGB, puntos originales e índices ganadores (`arg`). Conservar `arg` evita
que un empate cambie el color asignado al punto ganador.

| Proyección completa aislada | Mediana | p95 |
|---|---:|---:|
| Dos ordenamientos | 66.14 ms | 82.36 ms |
| Reducciones indexadas | 43.39 ms | 54.05 ms |

Reducción de 22.76 ms, **34.4%** en la mediana de proyección. No es una
medición aislada del z-buffer ni implica 34.4% más FPS del pipeline.

Evidencia: `zbuffer_equivalence_20261010/summary.json`, `frames.jsonl` y
log adjunto. El resumen conserva hashes del replay y del código de proyección.

## Pruebas

- Seis pruebas nuevas: vacío; colisiones con empates; signo del cero;
  NaN e infinitos; colisiones aleatorias float32/float64 hasta 307,200
  puntos; y todos los puntos sobre un mismo píxel.
- Las tres comprobaciones geométricas existentes pasan.
- La prueba del resumen temporal pasa, incluida identificación de la variante
  de z-buffer y compatibilidad con telemetría anterior. Diez comprobaciones
  en total y compilación Python sin errores.
- El pipeline dispone de variante anterior y nueva con la misma carga
  FS4/tracking5/escala0.5 para separar esta ganancia de cambios anteriores.

## Reproducción

La optimización está activa por defecto; el comando FS4 habitual no cambia.
Los comandos completos de equivalencia y A/B están en `../pose/REPLAY.md`.
Los resultados de sesiones previas se conservan: la ganancia válida es
contra la referencia contemporánea del mismo A/B.
