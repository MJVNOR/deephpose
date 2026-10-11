# Comparación controlada de tracking

Ejecutar desde la raíz del proyecto. Esta ruta offline complementa el modo
`live_pose.py --bag`: no avanza mientras cargan modelos ni mientras SAM o
register trabajan. No necesita cámara ni ventana.

## 1. Congelar la secuencia y la profundidad

```powershell
uv run python scripts/prepare_replay.py --bag data/bags/live_20261010_030134.db3 --out data/processed/replay_fixed --start-frame 60 --stride 15 --scale 0.5 --valid-iters 16
```

Selecciona números IR absolutos 60,75,90… (aproximadamente 2 Hz de la fuente
de 30 FPS). Verifica que no falten IDs intermedios ni el inicial. Primero
extrae las imágenes; después calienta FoundationStereo y prepara los bundles
RGB rectificado/profundidad/K con la misma geometría usada por live_pose.
`manifest.json` solo aparece al completar todos los bundles y contiene
IDs, timestamps, parámetros y hashes. Guardarlo junto a los bundles: es la
entrada fija para todas las variantes, independientemente del scheduling
del SDK en una extracción futura. Una extracción con IDs faltantes aborta.

## 2. Comparar 1, 2 y 5 iteraciones

```powershell
uv run --project pose python pose/benchmark_replay.py --replay data/processed/replay_fixed --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --out reports/replay_comparison --track-iters 1 2 5 --repeats 2 --mask-text 'computer mouse'
```

Comprueba los hashes y carga los mismos bundles en RAM. SAM y register usan
el primer bundle; se conserva la misma pose centrada para iniciar todas las
variantes. Los calentamientos se hacen sobre el primer frame y se restaura
el estado antes de empezar la secuencia. Las repeticiones alternan el orden
1→2→5 y 5→2→1 para reducir el sesgo de orden. Semillas fijas no garantizan
resultados CUDA idénticos bit a bit; se conservan todas las poses para
comprobar la variación observada.

Salidas: manifiesto de ejecución, máscara/pose/overlay inicial, JSONL por
variante, resúmenes y overlays en los mismos frames. Las carpetas de salida
deben ser nuevas: nunca se sobrescriben resultados anteriores.

Los umbrales de LOST son cobertura <30% o residuo >20 mm, como en el vivo;
un residuo ausente/no finito también cuenta como LOST. No se reinicializa
automáticamente. No se añaden umbrales de salto de pose porque penalizarían
el movimiento real y cambiarían la comparación con live_pose.

**Los Hz reportados corresponden solo a tracking offline.** FS está cacheado,
no compite por GPU, y se excluyen carga, métricas, visualización y escritura
del tiempo de `track_pose`. No equivalen a FPS en vivo. La calidad se compara
sobre los mismos saltos temporales del video, y los residuos son ajuste a
la profundidad, no exactitud contra una referencia externa.

Comprobaciones rápidas:

```powershell
uv run python tests/test_replay.py
```

## Barrido de FoundationStereo y reproducción a tiempo real

`scripts/compare_fs_replay.py --source <replay_fixed> --out <salida_nueva>`
reutiliza exactamente los pares IR guardados, alterna el orden de FS4/8/16
por frame y crea tres manifests con sus respectivos bundles y tiempos.
`pose/run_fs_comparison.py` ejecuta dos rondas de tracking=5 sobre cada uno,
usando `--initial-state <resultado_benchmark_replay>` para conservar la pose
y máscara iniciales del ensayo de referencia. Sus argumentos también
incluyen `--source`, `--out` y `--mesh`.

Para comparar el pipeline concurrente sobre el bag completo:

```powershell
uv run --project pose python pose/run_realtime_comparison.py --bag data/bags/live_20261010_030134.db3 --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --benchmark-init data/processed/replay_20261010_fixed_v3 --initial-state reports/replay_comparison_20261010_v2 --candidate 8 --out reports/realtime_nueva
uv run python scripts/summarize_realtime.py reports/realtime_nueva
```

Este ensayo pausa el bag durante la carga y calentamiento de ambos modelos,
restaura la pose común, vuelve al inicio y descarta hasta el frame inicial.
Después reproduce a velocidad real con las colas de frames recientes de
live_pose. Los frames procesados pueden variar entre pasadas: la prueba
offline compara precisión sobre entradas fijas; esta mide comportamiento
concurrente, latencia y descarte. Se comprueba que bag/mesh/estado pertenecen
a la misma referencia mediante hashes.

El runner ejecuta 5 pasadas en serie: referencia FS16 con costos anteriores,
dos optimizadas, otra referencia, y una del candidato con costos anteriores.
`--legacy-costs` conserva IPC de archivos y doble render solo para aislar su
impacto. La transferencia redundante a GPU ya está eliminada en ambos modos.
Las pasadas son headless, con render del overlay por pose pero sin ventana
de video a 30 Hz. Excluyen SAM/registro de las métricas de seguimiento porque
restauran un estado común; no miden el tiempo de arranque.

El vivo ahora usa IPC binario con IDs y timeout, sin PNG/NPY temporales;
reutiliza un render de profundidad para métricas y overlay. Las opciones
`--metrics-jsonl <archivo_nuevo>` y `--headless` también pueden usarse en una
ejecución normal. La inicialización asíncrona conserva el bundle de SAM
para que máscara, RGB y profundidad correspondan al mismo frame.
El replay normal también pausa el bag durante la carga de modelos, SAM y
registro, y lo reanuda al obtener la pose inicial. Así el seguimiento no
salta los segundos consumidos por la inicialización. En modo headless,
si no se obtiene máscara o falla el registro, termina con un error explícito.

## Validación FS4 contra FS8 y perfil concurrente

Ambas variantes usan el pipeline optimizado. El orden ABBA equilibra parte
de la variación temporal; cada pasada reproduce todo el bag con el mismo
estado inicial, sin otras pruebas GPU simultáneas:

```powershell
uv run --project pose python pose/run_realtime_comparison.py --bag data/bags/live_20261010_030134.db3 --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --benchmark-init data/processed/replay_20261010_fixed_v3 --initial-state reports/replay_comparison_20261010_v2 --reference 8 --candidate 4 --optimized-only --out reports/fs4_vs_fs8_nueva
uv run python scripts/analyze_fs_validation.py reports/fs4_vs_fs8_nueva --offline reports/fs_pose_comparison_20261010 --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj
```

La telemetría incluye tiempos de espera en colas, roundtrip del worker,
conversión a profundidad, proyección, tracking, render, métricas y overlay.
Son tiempos de pared bajo concurrencia (FS también expone CUDA events),
no tiempos de GPU aislada ni porcentajes aditivos del frame. Los resúmenes
de etapas excluyen INIT e incluyen TRACK y LOST para no sesgar el resultado.

Para revisar otra grabación con GUI, SAM normal y muestras visuales:

```powershell
uv run --project pose python pose/run_visual_validation.py --bag data/bags/live_20261010_122918.db3 --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --iters 8 4 --out reports/segunda_escena_nueva
uv run python scripts/summarize_realtime.py reports/segunda_escena_nueva
```

Este segundo ensayo inicializa SAM por separado; no sustituye al A/B con
pose común. `--snapshots <carpeta_nueva>` guarda el primer overlay y uno
cada 20 resultados, sobre el RGB usado por pose; la ventana habitual muestra
ese mismo par sincronizado de RGB y pose. Escribir muestras añade
costo: no usarlo en la comparación principal de velocidad.

## Proyección sin doble ordenamiento

El z-buffer normal usa dos reducciones indexadas: mínima profundidad por
píxel y primer índice original entre los empates. Conserva el índice ganador
para asignación de color y los vacíos `inf/-1`. Datos no finitos usan la ruta
anterior para conservar su semántica. `--sorted-zbuffer` permite repetir
la referencia anterior sin cambiar FS ni tracking.

```powershell
uv run python scripts/validate_zbuffer_replay.py --replay data/processed/replay_20261010_fixed_v3 --out reports/zbuffer_equivalence_nueva
uv run --project pose python pose/run_realtime_comparison.py --bag data/bags/live_20261010_030134.db3 --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --benchmark-init data/processed/replay_20261010_fixed_v3 --initial-state reports/replay_comparison_20261010_v2 --candidate 4 --compare-zbuffer --out reports/zbuffer_concurrent_nueva
uv run python scripts/analyze_fs_validation.py reports/zbuffer_concurrent_nueva --group-by sorted_zbuffer
```

La primera prueba ejecuta ambas proyecciones sobre la misma profundidad FS4
de cada frame y compara todos los arrays devueltos. El A/B hace cuatro
pasadas completas: anterior/nueva/nueva/anterior. No usar `--legacy-costs`
en esta comparación, porque también cambiaría IPC y render. La telemetría
y los hashes incluyen la variante de z-buffer y `project_to_rgb.py`.

## Perfil interno y comparación de escalas

```powershell
uv run python scripts/profile_fs_modules.py --raw data/processed/replay_20261010_fixed_v3/raw/000060.npz --out reports/fs_profile_nuevo
uv run python scripts/compare_scale_replay.py --source data/processed/replay_20261010_fixed_v3 --out data/processed/replay_scales_nuevo --pose-reference reports/fs_pose_comparison_20261010/fs4/iters5_repeat1/frames.jsonl --initial-state reports/replay_comparison_20261010_v2 --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj
uv run --project pose python pose/run_fs_comparison.py --source data/processed/replay_scales_nuevo --out reports/scale_pose_nuevo --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --initial-state reports/replay_comparison_20261010_v2 --variants scale0.5 scale0.4
uv run --project pose python pose/run_realtime_comparison.py --bag data/bags/live_20261010_030134.db3 --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --benchmark-init data/processed/replay_20261010_fixed_v3 --initial-state reports/replay_comparison_20261010_v2 --candidate 4 --compare-scales --reference-scale 0.5 --candidate-scale 0.4 --out reports/scale_concurrent_nuevo
uv run python scripts/analyze_fs_validation.py reports/scale_concurrent_nuevo --group-by scale
```

Ejecutar en serie para no hacer competir las pruebas GPU. El perfil usa
eventos CUDA por módulo; comprobar `cuda_kernel_trace_available` antes de
interpretar las trazas como kernels nativos. La comparación de profundidad
incluye región del objeto, bordes y desacuerdo de validez. La prueba offline
fija los frames y la prueba concurrente mide la cadencia real: ninguna
sustituye a la otra.

**Escala 0.4 no se recomienda como reemplazo equivalente de 0.5**: en esta
grabación perdió el seguimiento offline a partir del frame 1230 en ambas
repeticiones. Mantener `--scale 0.5 --valid-iters 4 --track-iters 5` para la
configuración recomendada; véase `../reports/escala_fs_display_20261010.md`.

La ventana ahora muestra RGB rectificado y pose del mismo bundle. Refrescar
la ventana no fabrica poses nuevas: el contenido avanza al ritmo del tracker.
Esto corrige la desalineación temporal, no el retraso del cálculo.

## Volúmenes estéreo equivalentes

`scripts/fs_volumes.py` normaliza las características una sola vez y construye
la concatenación con índices vectorizados. Conserva el borde izquierdo completo
del volumen de concatenación, tal como lo hace FoundationStereo. No cambia pesos,
resolución ni iteraciones; los archivos de `third_party` permanecen intactos.
El camino optimizado se activa al cargar FS. `--reference-volumes` en `live_pose.py`
restaura la implementación original para comparación. En herramientas de FS sin
ese argumento se puede usar la variable `DEEPH_FS_REFERENCE_VOLUMES=1`.

```powershell
uv run python scripts/validate_fs_volumes.py --replay data/processed/replay_20261010_fixed_v3 --out reports/volume_equivalence_nueva
uv run --project pose python pose/run_realtime_comparison.py --bag data/bags/live_20261010_030134.db3 --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --benchmark-init data/processed/replay_20261010_fixed_v3 --initial-state reports/replay_comparison_20261010_v2 --candidate 4 --compare-volumes --out reports/volume_concurrent_nueva
uv run python scripts/analyze_fs_validation.py reports/volume_concurrent_nueva --group-by reference_volumes
```

La primera prueba alterna ambos caminos sobre los 161 pares originales, verifica
igualdad exacta de disparidad y mide volúmenes con características reales. La segunda
ejecuta referencia/optimizado/optimizado/referencia con pose inicial común,
escala 0.5, FS4 y tracking5. En el resumen, `True` es referencia y `False` optimizado.

## Proyección y overlay del vivo

El vivo pide `return_details=False` a la proyección: conserva profundidad, máscara,
intrínsecos y RGB, pero omite la nube auxiliar usada por la exportación estática.
Los grids de píxeles son inmutables y se reutilizan por tamaño. Los ejes opacos se
dibujan directamente en RGB con las mismas coordenadas y antialiasing; no necesitan
máscaras ni mezclas de transparencia por toda la imagen.

`--reference-cpu` usa la proyección detallada y el dibujo anterior para A/B.
La ruta de exportación conserva los detalles por defecto. El comando normal de
cámara no cambia.

```powershell
uv run python scripts/validate_projection_overlay.py --replay data/processed/replay_20261010_fixed_v3 --out reports/cpu_equivalence_nueva
uv run --project pose python pose/run_realtime_comparison.py --bag data/bags/live_20261010_030134.db3 --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --benchmark-init data/processed/replay_20261010_fixed_v3 --initial-state reports/replay_comparison_20261010_v2 --candidate 4 --compare-cpu --out reports/cpu_concurrent_nueva
uv run python scripts/analyze_fs_validation.py reports/cpu_concurrent_nueva --group-by reference_cpu
```

El validador usa las poses FS4 ya guardadas en `reports/fs_pose_comparison_20261010`,
la pose inicial de `reports/replay_comparison_20261010_v2` y la malla del ratón
para situar los ejes en el centro del objeto, igual que el vivo.

## Tracking con rejilla compartida y caché CUDA

La ruta normal de `track_pose` mantiene la pose inicial en GPU y reutiliza, dentro
de cada recorte, la rejilla geométrica que RGB y XYZ comparten. Conserva los modos
bilinear y nearest originales. También evita vaciar la caché del allocator CUDA
al finalizar cada llamada de tracking. Las cinco iteraciones permanecen intactas.

`pose/tracking_fast.py` adapta una copia del predictor instalado: valida y retira
únicamente sus dos llamadas `empty_cache`, y cambia el constructor de recortes
solo en los globals de esa copia. No modifica los módulos originales ni la función
global `torch.cuda.empty_cache`. El registro inicial conserva sus vaciados de caché,
con una adaptación independiente descrita abajo; el tracking con visualizaciones
de debug conserva la ruta original.

`--reference-tracking` restaura el tracking anterior en `live_pose.py`; en la API,
usar `track_pose(..., reference=True)`. La rejilla se descarta al salir de cada
recorte, y un cambio en la estructura esperada del predictor produce un error
explícito para revisión.

```powershell
uv run --project pose python pose/profile_tracking.py --out reports/tracking_profile_nuevo
uv run --project pose python pose/profile_tracking.py --paired-only --out reports/tracking_paired_nuevo
uv run --project pose python pose/validate_tracking_fast.py --out reports/tracking_equivalence_nueva
uv run --project pose python pose/run_realtime_comparison.py --bag data/bags/live_20261010_030134.db3 --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --benchmark-init data/processed/replay_20261010_fixed_v3 --initial-state reports/replay_comparison_20261010_v2 --candidate 4 --compare-tracking --out reports/tracking_concurrent_nueva
uv run python scripts/analyze_fs_validation.py reports/tracking_concurrent_nueva --group-by reference_tracking
```

Ejecutar estas pruebas en serie. El perfil y la equivalencia usan los bundles FS4
de `data/processed/replay_fs_sweep_20261010/fs4`, la malla del ratón y el estado
inicial guardado. La equivalencia recorre cuatro secuencias independientes completas
y compara las matrices finales de cada frame, sin sustituirlas por poses de referencia
durante el seguimiento. La telemetría del vivo incluye `fp_reserved_mib`.

## Inicialización sin competencia de profundidad

Durante SAM y el registro se suspenden nuevas peticiones a FoundationStereo;
una petición ya enviada puede terminar. La captura continúa conservando el frame
más reciente y la ventana muestra el frame usado para inicializar. La inferencia
de profundidad se reanuda al terminar, si SAM no detecta el objeto o ante un error.

`initialization_fast.py` comparte la rejilla de recorte dentro de los predictores
de refinamiento y puntuación. Mantiene las 252 hipótesis, cinco iteraciones y los
vaciados de caché originales. Restaura ambos métodos al salir, también si falla.
El HUD distingue preparando, refinando y seleccionando.

La carga de malla calcula su diámetro en bloques con `mesh_initialization.py`,
evitando la matriz completa de distancias. Mantiene muestreo y resultado; se
aplica también al modo de referencia para evitar fallos de asignación de RAM.

`--reference-init` desactiva ambas mejoras para A/B. En la API,
`estimate_pose(..., reference=True)` usa el registro original. Las comparaciones
de tracking con estado inicial guardado no miden este cambio.

```powershell
uv run --project pose python pose/profile_initialization.py --out reports/init_profile_nuevo
uv run --project pose python pose/validate_initialization.py --out reports/init_equivalence_nueva
```

El segundo comando compara pose, 252 candidatos y puntuaciones en tres frames,
y después ejecuta FS en otro proceso sobre IR grabado para medir la competencia.
Usa máscaras fijas (SAM en el primer frame y proyecciones CAD en los otros dos),
por lo que no mide carga ni inferencia de SAM, ni exactitud frente a ground truth.
Ver [resultados](../reports/optimizacion_init_20261010.md).

## Modo experimental próximo a un segundo

```powershell
uv run --project pose python pose/live_pose.py --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --scale 0.5 --valid-iters 4 --track-iters 5 --mask-text 'computer mouse' --fast-init
```

`--fast-init` precarga SAM y lo mantiene residente. Prepara también el registro
con una entrada sintética que se descarta, restaurando el estado anterior del
estimador; ninguna pose sintética se publica. Esta preparación se muestra y se
mide por separado: no se promete iniciar el proceso completo en un segundo.

La búsqueda selecciona 42 orientaciones separadas en SO(3), hace una iteración
en cada una, las puntúa y refina cuatro veces las ocho mejores. Después puntúa
los finalistas y comprueba cobertura válida >= 0.3, residuo mediano <= 0.02 m
e IoU entre silueta CAD y máscara SAM >= 0.5. Si no cumple, ejecuta la búsqueda
completa. Estos criterios detectan fallos gruesos; no garantizan orientación
correcta, exactitud milimétrica ni ausencia de ambigüedad.

El modo habitual permanece disponible omitiendo `--fast-init`. No se puede
combinar con `--reference-init`. SAM conserva el modelo, resolución y umbral
originales; no se reutiliza una máscara vieja. Sin detección se mantiene el
reintento de SAM y no se inventa una pose. El fallback y los reintentos pueden
superar un segundo. La telemetría incluye `initialization` con tiempos de
preparación, SAM, registro, total y decisión de fallback.

```powershell
uv run --project pose python pose/profile_sam_startup.py --out reports/sam_profile_nuevo
uv run --project pose python pose/benchmark_init_shortlist.py --out reports/init_shortlist_nuevo
uv run --project pose python pose/benchmark_fast_startup.py --out reports/fast_startup_nuevo
```

Ejecutar en serie. El segundo comando compara máscaras guardadas por el perfil
original en `reports/sam_one_second_profile_20261010`; los otros dos usan RGB
grabado y SAM real. El tercero mantiene FS cargado pero sin inferir durante INIT,
igual que la pausa del vivo; excluye la adquisición y el cálculo del primer RGB-D.

## Recuperación automática

`--mask-text` activa recuperación tras tres ajustes inválidos consecutivos.
Captura y ventana tienen sus propios hilos; la búsqueda SAM corre en el executor
y el tracking continúa durante esa búsqueda. Solo se permite una búsqueda a la
vez, con espera de un segundo tras resultado vacío/error. Dos ajustes válidos
consecutivos invalidan una búsqueda pendiente para evitar reiniciar con un
resultado que ya no hace falta. `r` también invalida resultados anteriores.

El registro usa exactamente el RGB y profundidad de la máscara. Durante ese
registro FoundationPose deja de trackear (comparte estimador) y FS cede la GPU;
la captura y la ventana continúan. Después se descartan bundles capturados antes
de terminar el registro, se refina en uno nuevo y se exige cobertura >= 0.3,
residuo finito <= 0.02 m y silueta no vacía. Solo entonces se publica `RECOVERED`.
Un fallo vuelve a activar búsqueda. La ventana muestra RGB actual sin CAD mientras
la pose no es válida; al recuperar, vuelve al par RGB/pose sincronizado.

La primera inicialización de un bag conserva la pausa reproducible. Las
recuperaciones posteriores no pausan la grabación. SAM permanece residente con
recuperación automática, incluso sin `--fast-init`, para evitar recargarlo.
`--no-auto-recover` conserva el modo de recuperación manual. Las máscaras de
archivo no se desplazan automáticamente: la recuperación requiere `--mask-text`.

Para inyectar una pérdida controlada, cambiar el script del comando de replay a
`pose/validate_live_recovery.py`. Perturba la pose del tracker en tres llamadas;
SAM, profundidad, métricas y registro siguen siendo reales. No es una prueba de
oclusiones reales ni demuestra ausencia de falsas aceptaciones por las métricas.
