# Optimización FS y pipeline concurrente — 2026-10-10

## Resultado principal

**25.4% más actualizaciones de pose** en el A/B controlado: de **2.52 a
3.16 poses/s**, manteniendo tracking=5 y escala=0.5. No hubo LOST en las
cinco reproducciones completas (1,165 actualizaciones en total).

| Configuración | Pasadas | Poses/s sostenidas | Latencia mediana | Latencia p95 | LOST |
|---|---:|---:|---:|---:|---:|
| FS16, IPC de archivos y doble render | 2 | 2.52 | 583 ms | 679 ms | 0/409 |
| FS8, IPC de archivos y doble render | 1 | 2.99 | 524 ms | 597 ms | 0/242 |
| FS8, IPC binario y render compartido | 2 | 3.16 | 534 ms | 591 ms | 0/514 |

Latencia: recepción del frame en el hilo de captura → publicación de pose y
overlay, no exposición del sensor → pantalla. Throughput calculado por
intervalos reales entre publicaciones, agregado por tiempo, excluyendo el
intervalo inicial. Las pasadas cubren frame IR 61–2465 y 80.77 s de fuente
después del warmup, hasta EOF.

La mayor parte de la mejora viene de bajar FS a 8 iteraciones. La ruta
binaria/render compartido añade aproximadamente 5.9% de throughput frente
a FS8 con costos anteriores, pero esa comparación tiene una sola pasada
de referencia. **No bajó su latencia mediana respecto a FS8 por disco**:
524→534 ms. Se observa más solapamiento/contención GPU (tracking tarda más
por llamada aunque salgan más poses por segundo). No atribuir todo el
25.4% exclusivamente al IPC.

## Condiciones del A/B

- RTX 4070 Laptop 8 GB; grabación live_20261010_030134.db3.
- Carga y calentamiento con bag pausado; pose inicial común restaurada.
  Después se vuelve al inicio y se descarta hasta el frame de referencia 60.
- Reproducción a velocidad real, FS y Pose concurrentes, colas de frames
  recientes. Los frames intermedios procesados varían con el scheduling.
- Sin ventana: incluye render/overlay por pose, pero no el dibujo GUI a
  30 Hz. No confundir con una medición completa de sensor/pantalla.
- SAM y registro se excluyen del A/B para evitar comparar distintos estados
  iniciales. Hay una prueba separada de arranque normal con SAM.
- Orden: FS16 anterior, FS8 optimizado, FS8 optimizado, FS16 anterior,
  FS8 anterior. Sin benchmarks GPU paralelos.
- `--legacy-costs` reproduce IPC de archivos y doble render. La transferencia
  redundante a GPU ya está eliminada en ambos modos; su resultado numérico
  se comprobó idéntico al anterior en 161 frames.
- No son cifras directamente comparables con la primera línea base de
  ~1.8 FPS: aquella consumía video durante el arranque y no tenía este control.

## Barrido offline previo: mismos 161 frames

FS se ejecutó en orden rotado por frame, tras dos calentamientos por
configuración. Los bundles y sus hashes quedan guardados. Tracking=5,
misma máscara/pose inicial de FS16, dos repeticiones por variante.

| Iteraciones FS | Modelo mediano | Modelo p95 | FS + proyección mediana | Diferencia de depth frente a FS16 | LOST en cada repetición |
|---|---:|---:|---:|---:|---:|
| 4 | 164 ms | 189 ms | 264 ms | 0.80 mm | 0/160, 0/160 |
| 8 | 197 ms | 237 ms | 298 ms | 0.32 mm | 0/160, 0/160 |
| 16 | 262 ms | 311 ms | 364 ms | referencia | 0/160, 0/160 |

La diferencia de profundidad es la mediana de las medianas por frame sobre
píxeles válidos conjuntos. **No es error absoluto**, ni evalúa específicamente
los bordes. Las poses de las dos repeticiones de cada variante coincidieron
exactamente en este entorno. Se revisaron overlays del frame 1260 para FS4
y FS8: ambos acompañan visualmente el ratón en esa muestra.

FS4 es un candidato más rápido en offline; en este cambio se validó FS8
a tiempo real por su menor desviación frente a FS16. No se afirma que FS4
sea igual de preciso o esté validado en producción.

## Cambios implementados

1. `scripts/fs_model.py`: cada imagen se transfiere a CUDA una sola vez;
   antes la izquierda se transfería también para consultar su shape.
2. `scripts/fs_transport.py` y `scripts/fs_worker.py`: protocolo binario
   por pipes con encabezado JSON, payload de arrays, ID de solicitud,
   comprobación de longitudes, timeout, propagación de EOF y cierre del
   proceso. No se escriben PNG/NPY por frame en el modo normal. El protocolo
   anterior se conserva para compatibilidad y A/B.
3. `pose/pose_model.py` y `pose/live_pose.py`: un solo render de profundidad
   CAD compartido entre fit_metrics y draw_overlay; salida visual conservada.
4. `pose/live_pose.py`: la inicialización asíncrona conserva el bundle que
   generó la máscara SAM y usa ese mismo RGB/depth al registrar y evaluar.
   La reinicialización borra la geometría antigua del display.
   El replay normal pausa la fuente durante carga, SAM y registro; mantiene
   el manejo de selección manual sobre el frame retenido y falla explícitamente
   en headless si no hay máscara o falla el registro.
5. Telemetría JSONL, manifiestos con hashes, modo headless y arranque
   controlado de replay para hacer comparaciones medibles. Salida al EOF.

Los defaults numéricos se conservaron. Las optimizaciones de IPC y render
están activas por defecto; FS8 se selecciona explícitamente:

```powershell
uv run --project pose python pose/live_pose.py --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --scale 0.5 --valid-iters 8 --track-iters 5 --mask-text 'computer mouse' --bag data/bags/live_20261010_030134.db3
```

Omitir `--bag` para cámara. La primera prueba de arranque normal detectó
un salto del frame 157 al 1355 durante SAM/registro: 1 INIT y 4 LOST
(`live_fs8_normal_smoke_20261010.*`). Se corrigió pausando también el bag
normal durante la inicialización. La prueba corregida, sin estado guardado
ni `--benchmark-init`, completó 1 INIT y 11 TRACK, cero LOST, y cierre normal.
Pasó del frame 59 al 61 al iniciar el seguimiento; evidencia en
`live_fs8_normal_fixed_20261010.*`. Esta prueba corta verifica el arranque;
las cinco reproducciones completas anteriores verifican el seguimiento.

## Verificación y evidencia

- 17 comprobaciones pasan: profundidad (3), proyección (3), selección de
  replay (4), transporte (6) y reutilización de render (1).
- Transporte probado con un subproceso real: múltiples solicitudes, lecturas
  cortas, EOF, paquetes truncados/grandes y timeout con terminación del worker.
- FS16 antes/después de eliminar la transferencia extra: diferencia máxima
  de profundidad **0 m**, ninguna diferencia de validez, en los 161 frames.
- Overlay: mismos píxeles, máscara y métricas al compartir un render; la
  prueba comprueba una llamada al render en vez de dos sin modificar RGB.
- Seis pasadas offline de pose y cinco reproducciones concurrentes completadas.

Archivos principales:

- `fs_sweep_summary_20261010.json`: barrido FS.
- `fs_pose_comparison_20261010/`: poses, overlays, manifiestos y logs offline.
- `fs16_equivalence_20261010.json`: equivalencia numérica.
- `realtime_optimized_20261010/aggregate.json`: resultados agrupados A/B.
- `realtime_optimized_20261010/metrics_summary.json`: resultados por pasada.
- `realtime_optimized_20261010/*.jsonl`: telemetría completa por pose.
- `realtime_optimized_20261010/*.manifest.json`: argumentos y hashes de código.
- `../pose/REPLAY.md`: instrucciones para repetir las pruebas.

Límites: una escena/objeto/GPU, dos repeticiones principales, sin ground
truth físico de pose o profundidad; LOST y residuos son métricas internas.
El modo GUI todavía superpone la última pose sobre RGB más reciente; eso
no convierte la frecuencia del video en frecuencia de actualización de pose.
