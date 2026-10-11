# Validación FS4, perfil concurrente y cámara — 2026-10-10

Continuación: la optimización del z-buffer identificada aquí se implementó
y evaluó después; véase `optimizacion_zbuffer_20261010.md`. Las cifras de
este informe corresponden al pipeline anterior a ese cambio.

## A/B completo FS4 contra FS8

**FS4 entrega 8.6% más poses/s que FS8 en el A/B de esta sesión**, con menor
latencia mediana y sin LOST. Se conservó escala 0.5 y tracking 5.

| Variante optimizada | Pasadas | Resultados | Poses/s agrupadas | Latencia p50 | Latencia p95 | LOST |
|---|---:|---:|---:|---:|---:|---:|
| FS8 | 2 | 540 | 3.326 | 517 ms | 568 ms | 0 |
| FS4 | 2 | 586 | 3.612 | 497 ms | 556 ms | 0 |

Cada pasada cubrió IR 61–2465 del video original. Resultados por orden:
FS8 3.355, FS4 3.650, FS4 3.574, FS8 3.298 poses/s. Agrupación por tiempo
real entre publicaciones; primera latencia excluida. Estas referencias FS8
son contemporáneas: no calcular la ganancia comparando directamente FS4
con los 3.16 poses/s de otra sesión.

| Etapa, mediana de pared | FS8 | FS4 |
|---|---:|---:|
| Espera desde captura | 18.5 ms | 18.8 ms |
| Roundtrip FS | 223.5 ms | 196.5 ms |
| Conversión a profundidad | 3.6 ms | 4.0 ms |
| Proyección a RGB | 71.4 ms | 73.4 ms |
| Espera del bundle | 0.26 ms | 0.27 ms |
| Tracking | 158.6 ms | 159.7 ms |
| Render CAD | 4.6 ms | 4.6 ms |
| Métricas de ajuste | 1.5 ms | 1.5 ms |
| Overlay | 29.4 ms | 30.4 ms |

## Diferencias de pose sobre entradas fijas

Sobre 160 frames idénticos del barrido offline FS4/FS8:

| Discrepancia entre estimaciones | Mediana | p95 | Máxima |
|---|---:|---:|---:|
| Orientación | 0.48° | 1.26° | 4.13° |
| Posición del centro del bounding box CAD | 0.28 mm | 0.73 mm | 3.09 mm |

No son errores contra ground truth. El frame 1335 tiene la mayor diferencia
angular; se compararon cajas sobre el mismo RGB y ambas rodean el ratón,
con diferencias de orientación visibles. La traslación del origen OBJ
alcanza 17.19 mm porque ese origen está desplazado del centro del objeto:
para interpretación física se reporta también el centro transformado.

## Alcance y método

Se amplió el benchmark de una a dos grabaciones del mismo ratón MX Master,
misma D415, malla y GPU RTX 4070 Laptop. Escala 0.5, tracking 5.
La segunda secuencia se capturó con el usuario moviendo y girando el ratón.
No representa diversidad de objetos, iluminación o cámaras.

1. Cámara real con ventana, FS8 y SAM normal: 120 resultados y grabación nueva.
2. Video original completo: FS8 / FS4 / FS4 / FS8, ambas variantes optimizadas,
   pose inicial común, fuente a velocidad real, sin ventana ni escritura de imágenes.
3. Segunda grabación: replay con ventana y SAM independiente para FS8 y FS4,
   conservando muestras visuales cada 20 resultados.
4. Comparación de poses offline sobre los mismos 160 frames, sin confundir
   discrepancia entre estimaciones con error absoluto.

## Cámara real

`camera_validation_fs8_20261010.jsonl` y `.summary.json`:

- 1 INIT y 119 TRACK, cero LOST, salida normal tras 120 resultados.
- 2.9565 poses/s entre la primera y última publicación (40.25 s).
- Latencia de captura recibida a publicación: mediana 516.6 ms, p95 616.2 ms.
- Nueva fuente: `../data/bags/live_20261010_122918.db3` (1,934,761,984 bytes).
- Se revisaron overlays de los frames 824, 1199, 1387, 1569 y 1754:
  objeto apoyado, levantado y girado; la silueta permanece sobre el ratón.
  La mano produce oclusión parcial. Las muestras no certifican tolerancia
  a oclusión total ni a todos los movimientos rápidos entre muestras.

La cámara física sigue avanzando durante SAM: el primer resultado usa el
frame 170 y el tramo de fuente entre primer/último resultado abarca 53.22 s,
frente a 40.25 s entre publicaciones. Esto confirma por qué se requiere
mantener el objeto quieto al inicializar. Un bag sí puede pausarse.

## Segunda grabación completa con ventana

Ambas pasadas cubrieron frames IR 57–1796, 58.425 s de fuente, con SAM
independiente y cierre por EOF. Incluyen ventana y muestras cada 20 poses;
no son la comparación aislada del A/B principal.

| Variante | Resultados | Poses/s | Latencia p50 | Latencia p95 | LOST | Residuo mediano interno |
|---|---:|---:|---:|---:|---:|---:|
| FS8 | 179 | 3.019 | 523 ms | 585 ms | 0 | 2.3 mm |
| FS4 | 197 | 3.324 | 496 ms | 571 ms | 0 | 2.7 mm |

FS4 aumentó throughput 10.1% en esta comparación de una pasada por variante.
El residuo interno subió ligeramente; no demuestra por sí solo un error
absoluto mayor. Se inspeccionaron las siluetas de FS4 en frames 1316 y 1658
y de FS8 en 1238, además de las muestras de cámara: siguen el objeto al
elevarlo y girarlo, con oclusión parcial por la mano. No se certifica oclusión
total ni precisión submilimétrica.

Evidencia: `second_scene_visual_20261010/metrics_summary.json`, JSONL,
logs y carpetas de imágenes. En esta ampliación: 1,126 resultados del A/B,
376 de replay con GUI y 120 de cámara; **1,622 resultados sin LOST**.

## Interpretación del perfil

Cada bundle transporta sus tiempos y la pose registra esperas, tracking,
render, cálculo de ajuste y overlay. La captura usa `perf_counter`; el
worker conserva los CUDA events de FS. No se añadieron sincronizaciones
CUDA extras al pipeline. Los tiempos de pared reflejan también contención.

La primera medición con cámara tuvo medianas de 230.6 ms en roundtrip FS,
74.4 ms en proyección CPU, 148.1 ms en tracking, 4.6 ms en render y 30.4 ms
en overlay. La espera de bundles fue 0.29 ms. FS y tracking se solapan:
sumar medianas no predice el throughput y puede contar trabajo concurrente.

El siguiente trabajo de rendimiento debe concentrarse en FS y proyección,
no en quitar otro render (ya cuesta pocos ms). La proyección reconstruye
rayos/calibración, rectificación y buffers cada frame y realiza ordenamiento
de profundidad y `unique` para el z-buffer. Son candidatos a optimización,
no ganancias demostradas: requieren comparación geométrica.

El perfil interno ya se ejecutó aisladamente sobre profundidad FS4 real
de frames 60, 1260 y 2460, diez repeticiones por entrada (30 llamadas):
mediana 67.90 ms, p95 70.36 ms, 2.004 s acumulados. `splat_to_rgb` ocupa
0.899 s (44.9%, 30.0 ms/llamada), incluidos dos argsort por llamada
(0.593 s, 29.6% del total, 19.8 ms/llamada). `depth_to_xyz` ocupa 0.268 s
(13.4%, 8.9 ms/llamada). La rectificación tarda 0.004 s acumulados en esta
calibración y no es el cuello de botella. Los tiempos inclusivos se solapan;
no sumar `unique`, argsort y `splat_to_rgb` como costos independientes.

Evidencia: `projection_profile_20261010/projection.txt`, `.prof` y
`summary.json`. Prioridad siguiente: sustituir el doble ordenamiento del
z-buffer conservando el píxel ganador, desempates y exactitud de proyección;
después reutilizar rayos/calibración. No se implementó ese cambio en esta
validación para mantener la comparación FS4/FS8 bajo el mismo pipeline.

## Configuración resultante

FS4 es una opción validada para estas dos secuencias del ratón. Conserva
tracking=5 y escala=0.5; usar FS8 como referencia si se amplían condiciones.
No se han validado objetos distintos ni se promete precisión absoluta.

```powershell
uv run --project pose python pose/live_pose.py --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --scale 0.5 --valid-iters 4 --track-iters 5 --mask-text 'computer mouse' --bag data/bags/live_20261010_030134.db3
```

Omitir `--bag` para cámara. Los FPS son actualizaciones de pose, no frames
del video. Bajar a FS4 no elimina el desfase visual de aproximadamente medio
segundo. Pasar a 10–30 poses/s requerirá cambios adicionales en el modelo
o en la frecuencia con que se calcula profundidad, con nueva validación.

## Limitaciones visuales

La ventana dibuja la última pose disponible sobre RGB más reciente; no es
una pose nueva a 30 Hz. Las muestras guardadas usan el RGB de la estimación
y permiten distinguir ajuste geométrico de desfase temporal. Un estado
TRACK solo significa que pasó los umbrales internos de cobertura/residuo.
No hay ground truth de pose ni medición desde exposición física a pantalla.

## Reproducibilidad

- `fs4_vs_fs8_concurrent_20261010/`: cuatro JSONL, logs, hashes de código y
  resumen agrupado; cada pasada debe alcanzar el final del video original.
- `fs4_fs8_pose_differences_20261010/`: cajas sobre RGB idéntico para los
  frames 1290, 1335 y 1710, seleccionados por alta discrepancia angular.
- `camera_validation_fs8_20261010_frames/`: muestras de cámara real.
- `../pose/REPLAY.md`: comandos para repetir las comparaciones.
- `second_scene_visual_20261010/source_manifest.json`: hashes de nueva
  grabación y malla.

Verificación de código: seis pruebas de transporte, una de reutilización
del render y una de resumen temporal pasan; compilación Python de los siete
archivos modificados/añadidos de ejecución y análisis, sin errores. La nueva
prueba del resumen verifica que INIT no contamine las etapas y que LOST
permanezca incluido en las estadísticas.

No se modificaron los defaults numéricos. Las nuevas opciones permiten
ABBA optimizado (`--optimized-only --reference 8 --candidate 4`), telemetría
por etapa y muestras visuales (`--snapshots`).
