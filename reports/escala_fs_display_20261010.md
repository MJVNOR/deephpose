# Escala FS, perfil interno y display sincronizado — 2026-10-10

## Decisión de calidad

**Mantener escala 0.5, FS4 y tracking5. No adoptar escala 0.4 como reemplazo
equivalente.** En el ensayo offline con frames fijos, 0.4 pierde tracking
en 83/160 frames en ambas repeticiones; 0.5 no pierde ninguno. Primera
pérdida en frame 1230. Las repeticiones de cada escala dan poses idénticas.
Esa pérdida persiste hasta el final: son 83 frames marcados LOST, no 83
episodios independientes. El benchmark conserva la política existente de
no reinicializar automáticamente durante el seguimiento.

La prueba usa exactamente los mismos 161 pares IR, RGB, calibración, malla,
máscara y pose inicial. FS se recalcula para cada escala alternando el orden;
tracking consume los 160 frames siguientes sin descartar ninguno. El paso
de fuente es aproximadamente 0.5 s, por lo que es una prueba de robustez a
desplazamientos entre frames distinta del pipeline a tiempo real.

## Comparación a tiempo real

Cuatro pasadas completas ABBA, todas con FS4, tracking5, z-buffer optimizado,
estado inicial común y modo headless; frames IR 61–2465 en cada pasada:

| Escala | Pasadas | Resultados | Poses/s agrupadas | Latencia p50 | Latencia p95 | LOST |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 2 | 591 | 3.647 | 500 ms | 551 ms | 0 |
| 0.4 | 2 | 683 | 4.215 | 444 ms | 484 ms | 0 |

**15.6% más poses/s con 0.4 en este A/B**, sin pérdidas en las pasadas
concurrentes. Esto no invalida la regresión offline: el pipeline más rápido
procesa frames más cercanos entre sí, mientras el ensayo fijo salta 15 frames
de origen entre actualizaciones. La mayor cadencia puede compensar cambios
de profundidad en esta secuencia; no demuestra igual tolerancia ante pausas,
sobrecarga o movimientos grandes. No hay ground truth de exactitud.

La recomendación sigue siendo 0.5 para priorizar robustez y medición.
0.4 queda como opción experimental de rendimiento con compromiso de
profundidad, no como mejora sin pérdida de calidad. Comparar contra las
referencias de esta sesión, no contra los 4.00 poses/s de otro A/B.

Evidencia: `scale_concurrent_20261010/validation_summary.json`, telemetría,
manifiestos y logs. La latencia mide recepción del frame a publicación;
no incluye exposición física ni presentación en pantalla.

En el ensayo offline, frame 1560, se revisaron los overlays de ambas escalas: 0.5 sigue al
ratón, mientras 0.4 dibuja el CAD desplazado y con orientación incorrecta.
No se trata solamente de cruzar un umbral numérico de LOST.

## Profundidad: rapidez frente a diferencias geométricas

FS4 aislado sobre el barrido de 161 frames: mediana 155 ms a escala 0.5
y 125 ms a escala 0.4. Son tiempos del modelo, no FPS del pipeline.

Para cada frame se midió diferencia absoluta frente a 0.5 en píxeles válidos
comunes. La tabla da **medianas entre frames** de sus percentiles por píxel:

| Región | Mediana de diferencia | Mediana del p95 por frame |
|---|---:|---:|
| Imagen completa | 2.67 mm | 20.12 mm |
| Bounding box proyectado del ratón | 2.22 mm | 14.16 mm |
| Bordes de profundidad dentro del bounding box | 8.09 mm | 72.60 mm |

La región del objeto se obtiene de la caja CAD proyectada con las poses
guardadas de 0.5; incluye fondo/mano, no es segmentación perfecta. Bordes:
diferencias >5 mm entre vecinos finitos, dilatadas un píxel. En discontinuidades,
un desplazamiento de borde puede comparar objeto con fondo y producir
grandes diferencias. En la región del objeto, mediana de desacuerdo de
validez 8.08%, p95 entre frames 10.87%. No se consideran equivalentes los
píxeles que una variante deja inválidos.

Estos números son discrepancias contra 0.5, **no errores contra ground truth**.
Tras perder tracking, la diferencia de pose escala0.4/0.5 alcanza una mediana
de 20.41° y 104.0 mm en el centro CAD. No usar esas poses para medir objetos.

Evidencia: `scale_quality_20261010/summary.json`, mapas de profundidad de
frames 1260/2250, `scale_pose_comparison_20261010/` con overlays, poses y
dos repeticiones; bundles y hashes en
`../data/processed/replay_scale_sweep_20261010/`.

## Perfil interno de FoundationStereo

Sobre el frame IR 60: tres warmups y cinco inferencias por escala con eventos
CUDA alrededor de hijos directos del modelo. Los módulos repetidos acumulan
sus llamadas en cada inferencia. La resolución tras padding pasa de
320×256 a 256×192 (40% menos píxeles). FS4 en esta muestra: 172→112 ms.
La diferencia respecto al barrido refleja entrada, calentamiento y carga;
no mezclar esas cifras como si fueran una sola medición.

| Bloque, intervalo CUDA mediano | Escala 0.5 | Escala 0.4 |
|---|---:|---:|
| Extracción de características (`feature`) | 32.9 ms | 23.3 ms |
| Refinamiento recurrente, cuatro llamadas (`update_block`) | 26.8 ms | 17.5 ms |
| Clasificador 3D (`classifier`) | 26.1 ms | 16.1 ms |
| Agregación del volumen (`cost_agg`) | 19.3 ms | 11.7 ms |
| Entrada del volumen (`corr_stem`) | 11.3 ms | 5.8 ms |
| Contexto (`cnet`) | 4.9 ms | 3.8 ms |

Hay trabajo funcional fuera de estos módulos, incluyendo construcción y
consulta del volumen y conversiones. No sumar medianas para atribuir todo
el tiempo ni confundir intervalos de stream con tiempo puro de kernels.

**Límite del profiler:** PyTorch 2.6.0+cu124 instalado declara solo actividad
CPU en `torch.profiler.supported_activities()`. Las trazas exportadas no
contienen eventos nativos de kernels CUDA; su tabla de operadores con tiempos
CUDA usa el mecanismo disponible y altera sustancialmente la ejecución.
Se conserva como evidencia exploratoria, no como presupuesto fiable por
kernel. Los tiempos de la tabla anterior usan eventos CUDA explícitos,
independientes de esa llamada al profiler. No se instalaron dependencias.

Evidencia: `fs_internal_profile_20261010/summary.json`, tablas de operadores
y trazas. Para acelerar conservando resolución, los candidatos son esos
bloques y la construcción/consulta del volumen; cualquier fusión/compilación
debe validarse sobre los mismos frames antes de adoptarse.

Se amplió el perfil con spans funcionales (otra sesión aislada, cinco
inferencias por escala), en `fs_internal_profile_detailed_20261010/`.
A escala 0.5, FS total mediano 153 ms; `build_gwc_volume` 19.81 ms,
`build_concat_volume` 5.54 ms y las cuatro consultas del volumen geométrico
6.63 ms. Otros bloques: características 28.56 ms, clasificador 26.26 ms,
refinamiento 22.91 ms y agregación 19.06 ms. A 0.4, total 124 ms.

La construcción de ambos volúmenes ocupa alrededor de 25 ms sin bajar
resolución: es un candidato concreto para vectorización/fusión preservando
las entradas y salidas. No se implementó ni se atribuyó una ganancia a ese
cambio en esta tarea. La instrumentación solo existe en el proceso de perfil;
los benchmarks de FPS se ejecutaron sin esos hooks y sin otra prueba GPU
simultánea. Esta segunda medición no reemplaza los datos de la primera.

## Corrección de ventana

`pose/live_pose.py` ahora muestra el overlay ya compuesto sobre el RGB
rectificado del mismo bundle usado por la pose. Ya no redibuja la caja con
intrínsecos rectificados sobre RGB crudo más reciente. También desaparece
ese trabajo extra de dibujo en cada refresco de ventana.

El contenido cambia cuando termina una pose; el bombeo de eventos sigue
activo para teclas y cierre. Antes de la primera estimación se muestra RGB
o la pantalla de inicio; durante SAM/registro se muestra la imagen retenida.
Al reinicializar se limpia el overlay antiguo. La imagen y la pose quedan
alineadas temporalmente, **pero no desaparece la latencia de procesamiento**.

La selección de imagen tiene pruebas para impedir que un RGB nuevo sustituya
a una estimación completa y para las rutas de arranque/vacío.

Se reprodujo además el video completo con la ventana corregida, escala0.5,
FS4/tracking5 y estado inicial común: **289 resultados, cero LOST**, frames
IR 61–2465, 3.566 poses/s y latencia mediana 510 ms. Esta pasada incluye
GUI y guardado de una muestra cada 20 resultados; no sustituye al A/B headless.
Se revisó el overlay del frame 1205 sobre su RGB correspondiente.
Evidencia: `synchronized_display_20261010.jsonl`, `.summary.json`, manifiesto,
log y `synchronized_display_20261010_frames/`. Salida normal al EOF.

Verificación final: tres pruebas de selección del display, una del resumen
temporal (incluye escala y variante) y una de reutilización del overlay
pasaron. Compilación Python de los nueve archivos de ejecución/análisis
modificados o añadidos, sin errores. No se cambiaron los defaults numéricos
ni los umbrales de LOST para mejorar artificialmente los resultados.
