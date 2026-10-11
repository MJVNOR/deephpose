# Evaluación del proyecto y FPS — 2026-10-10

## Alcance y evidencia

Revisión de los 20 archivos Python propios de scripts/, pose/ y tests/, configuración, documentación, logs y rutas críticas de las dependencias locales FoundationStereo/FoundationPose. No es una auditoría exhaustiva de todas las librerías vendorizadas. No se modificó el código de aplicación ni los cambios preexistentes del usuario. No se abrió la cámara.

- Sintaxis: 20 archivos correctos.
- tests/test_depth.py: 3 checks correctos; se repitió fuera del sandbox por bloqueo de archivos temporales.
- tests/test_project.py: 3 checks correctos.
- GPU detectada: RTX 4070 Laptop, 8188 MiB. Uso ajeno/desktop variable; las mediciones no son un laboratorio aislado.
- live_run.log: 529 resultados, mediana de FPS registrados 2.8, t_pose 0.03 s, latencia 0.3 s; **528 LOST**. No representa seguimiento válido a 2.8 FPS.
- live_sam5.log: solo tres resultados TRACK (1.6, 2.1, 2.1 FPS), t_pose 0.11/0.04/0.05 s. INIT t_pose=50.42 s. Muestra demasiado corta para estabilidad sostenida.
- README/plan registran FS completo ~0.64–0.66 s con scale=1, 16 iteraciones. No consta la configuración exacta de todos los logs live: no comparar directamente esas ejecuciones.

## Benchmark ejecutado en esta revisión

Captura 20261008_040430_emitter-on, FS aislado, 2 calentamientos por configuración y 5 inferencias medidas. Medianas del wrapper completo (preparación, modelo y retorno CPU); no incluyen proyección, pose ni display. Orden fijo, una sola escena, GPU con carga de escritorio variable.

| Escala | Iteraciones | Total mediano | Inferencias/s equivalentes | Diferencia mediana de profundidad frente a 1.0/16 |
|---|---:|---:|---:|---:|
| 1.0 | 16 | 802 ms | 1.25 | referencia |
| 1.0 | 8 | 611 ms | 1.64 | 0.12 mm |
| 0.5 | 8 | 298 ms | 3.36 | 1.77 mm |
| 0.5 | 4 | 252 ms | 3.97 | 1.78 mm |

Escala 0.5/8 acelera esta etapa 2.69×; 0.5/4, 3.18×. La diferencia se mide contra la salida del propio FS a resolución completa, remuestreada nearest, sobre validez conjunta: **no es error absoluto ni validación de pose/bordes**. El benchmark confirma que bajar solo iteraciones no reduce todo el costo proporcionalmente. Datos individuales en fps_audit_benchmark_20261010.json. Estos valores no garantizan la misma mejora al ejecutar Pose y FS juntos.

## Hallazgos de corrección y robustez

1. **Se mezclan tiempos y geometrías en el display.** live_pose.py:268 usa el RGB crudo más reciente con K_rect y una pose calculada sobre otro frame. Con distorsión no nula, hay además diferencia de geometría. Mostrar frame rectificado y pose correspondientes; un modo de video fluido con pose anterior debe mostrar su antigüedad. view.geom prevalece sobre view.vis, incluso tras r: queda visible la caja anterior durante reinicialización.
2. **SAM pierde la identidad del frame.** live_pose.py:314 envía un rect a SAM, pero al completar usa la máscara con el rect/buf de un bundle posterior (línea 362). Si el objeto se mueve, el registro empieza con máscara incorrecta. Conservar el bundle completo con cada future y su resultado. El resultado de register también se evalúa con el bundle corriente en vez del que se registró.
3. **LOST no recupera el objeto.** live_pose.py:394 solo cambia la etiqueta y continúa desde pose_last. El registro largo demuestra pérdida persistente. Añadir estados de recuperación con umbrales, histéresis, intentos limitados y segmentación nueva; evitar repetir búsquedas globales sin pausa.
4. **El timeout SAM no limita la espera real.** live_pose.py:70 usa un executor dentro de with. Al salir tras TimeoutError, el context manager espera al trabajo activo. shutdown(wait=False) tampoco cancela un trabajo ya iniciado. Un proceso aislado permite terminar una inferencia colgada; como mínimo, tratar timeout y cancelación explícitamente. Ver documentación Python enlazada al final.
5. **Errores de hilos y salida.** ask_worker bloquea en readline sin timeout ni request_id; excepciones en infer_loop pueden dejar main esperando indefinidamente. q solo se consume al recibir un bundle. pool se crea tarde pero se usa siempre en finally: un fallo temprano puede quedar oculto por UnboundLocalError. Inicializar recursos antes del try, propagar errores y bombear controles aunque no lleguen datos.
6. **--emitter no se aplica en live_pose.py.** Se declara pero no se configura el sensor. También abre el stream depth sin consumirlo. Aplicar y registrar el estado real; usar depth para modo rápido o desactivar el stream cuando no haga falta.
7. **IPC frágil.** split() rompe rutas con espacios; stdout del modelo puede confundirse con respuestas; _live_ring fijo colisiona entre instancias. Mantener procesos separados por la colisión Utils es razonable. Usar mensajes estructurados, stderr para logs, IDs y memoria compartida con propiedad explícita de buffers.
8. **Geometría no completamente validada.** rectify_color trata inverse_brown_conrady y modified_brown_conrady como el modelo Brown estándar; necesitan validación específica frente al SDK cuando los coeficientes son no nulos. Los tests actuales cubren identidad, no esos casos. La carga de Scene concatena geometrías sin aplicar transformaciones de nodos; afecta escenas que las contengan. No afirmo que el OBJ actual las tenga.
9. **Evaluación incompleta.** evaluate.py falla con percentiles si no hay píxeles válidos conjuntos. Tests no cubren cola, IPC, cancelación, SAM, pérdida de pose, reinicio ni movimiento. El residuo contra la misma profundidad usada para estimar no prueba exactitud absoluta. Faltan dimensiones físicas confirmadas de la malla y referencia externa de pose/distancia.
10. **Reproducibilidad.** live_pose mantiene defaults separados de config.toml. README no describe el estado actual de pose; extensiones compiladas y repos externos requieren receta y revisiones exactas para reconstruir el entorno. pesos_dir de make_estimator no se utiliza.

## Costos evitables encontrados

- **FS en cada bundle** limita la frecuencia de profundidad/pose; más hilos no eliminan el trabajo de una GPU compartida. Medir FS aislado, pose aislada y ambos juntos.
- **PNG/NPY por frame**: live_pose.py:227–232, fs_worker.py. Compresión, disco, decodificación y copias interprocesos. Sustituir con buffers de memoria compartida y control de secuencia.
- **Render redundante**: fit_metrics renderiza CAD y draw_overlay vuelve a renderizarlo (pose_model.py:116,160). En el live, ese overlay queda descartado visualmente cuando existe view.geom. Reutilizar un render o prescindir del overlay costoso en modo caja.
- **empty_cache en cada predict**: FoundationPose/learning/training/predict_pose_refine.py:237. Evaluar quitarlo del tracking estable y conservar liberación explícita al descargar SAM o cambiar de etapa. Medir pico conjunto antes de decidir: hay dos procesos CUDA y 8 GB.
- **Transferencia duplicada**: fs_model.py:56 ejecuta mk(img0) solo para obtener shape y lo repite en la siguiente línea. Usar el tensor ya creado o construir la forma en CPU.
- **Proyección CPU completa**: mallas de coordenadas, matrices, buffers float64 y argsort+unique por frame. Cachear rayos y mapas de rectificación; usar float32 donde la precisión medida lo permita; ruta min-Z sin índices si el consumidor solo necesita profundidad. Preservar el z-buffer y probar oclusiones.
- **Instrumentación y logs en ruta crítica**: reset_peak_memory_stats y eventos nuevos por frame, múltiples sincronizaciones, logs upstream por iteración. Hacer profiling configurable y logs agregados. No quitar sincronizaciones de los benchmarks: produciría tiempos GPU incorrectos.
- **live.py calcula FS aun con --view d415** y genera ambos colormaps. Si se necesita visualizar solo D415, ofrecer ruta que omita FS explícitamente.

## Orden recomendado

1. Corregir coherencia frame/máscara/pose, recuperación LOST y salida. Registrar FPS de captura, profundidad, TRACK válido y display por separado; latencia p50/p95 desde adquisición hasta publicación, drops e inicialización aparte. t_age actual solo mide espera antes de FS; no es antigüedad de la pose dibujada.
2. Benchmark reproducible en replay: misma secuencia, warmup, 100–300 frames, configuración y revisiones registradas; movimiento lento/rápido, oclusión y retorno. Guardar calidad, cobertura, jitter, pérdidas y VRAM total además de FPS.
3. Probar scale=0.5 y valid_iters=8; después 4. Probar track_iters=1. Son compromisos de precisión, no mejoras gratuitas. Comparar bordes y pieza además de la escena global.
4. Quitar render duplicado, transferencia duplicada y logs intensivos; probar política de caché de Pose. Después sustituir IPC de disco y cachear proyección. Medir cada cambio por separado.
5. Si hacen falta 15–30 actualizaciones reales/s, evaluar una ruta D415→alineación/rectificación coherente→FoundationPose. FS puede quedar para inicialización o correcciones periódicas. No combinar RGB nuevo con profundidad FS vieja sin compensación temporal. La calidad de profundidad D415 sobre la pieza decide si sirve esta arquitectura.
6. Considerar reducción de malla conservando apariencia y silueta, o compilación/exportación del modelo, solo después de perfilar y comprobar compatibilidad del stack Windows instalado. No prometer 30 FPS con el pipeline actual.

## Referencias verificadas

- Python 3.11, Executor/ThreadPoolExecutor: https://docs.python.org/3.11/library/concurrent.futures.html
- PyTorch, semántica CUDA y caché: https://docs.pytorch.org/docs/main/notes/cuda.html

Las ganancias de las optimizaciones propuestas siguen pendientes de medición end-to-end. El benchmark FS separado completó y está guardado en fps_audit_benchmark_20261010.json.
