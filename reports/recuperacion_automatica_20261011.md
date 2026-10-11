# Recuperación automática y revisión de SAM 3.1

## Cambios

Con `--mask-text`, tras tres poses inválidas consecutivas se inicia una búsqueda
SAM asíncrona mientras FoundationPose sigue intentando trackear. Solo se permite
una búsqueda pendiente. Dos ajustes válidos consecutivos o un reinicio manual
invalidan resultados pendientes. Resultado vacío/error reintenta con backoff de
un segundo; un worker fallido se recrea en el siguiente intento.

La nueva máscara registra sobre su RGB-D original. Captura y ventana continúan
durante el registro, mostrando imagen actual sin CAD antiguo. El estimador no
registra y trackea simultáneamente; FS cede temporalmente la GPU al registro.
Antes de aceptar la recuperación se exige un frame capturado después del fin
del registro, refinamiento y ajuste válido: silueta no vacía, cobertura >= 0.3,
residuo finito <= 0.02 m. Entonces se publica `RECOVERED`.

No aumenta por sí solo los FPS del tracking ni garantiza cero pérdidas. Los
criterios geométricos tampoco demuestran identidad correcta o pose ground truth.
El SAM elegido permanece residente durante recuperación automática. Se puede
desactivar con `--no-auto-recover`; las máscaras estáticas no habilitan esta función.

## Validación

La prueba `pose/validate_live_recovery.py` perturba la posición X del estimador a
2 m en las llamadas de tracking 4, 5 y 6. SAM, FS, registro y métricas son reales;
no se falsifican sus resultados. Usa el bag `live_20261010_030134.db3`, escala
0.5, FS 4 iteraciones, tracking 5 y `--fast-init`. Las tres ejecuciones terminaron
con exit code 0, 30 poses, cuatro estados LOST, recuperación en la pose 9 y
21 poses posteriores sin pérdidas. No equivale a probar oclusiones reales.

| Ejecución | Primer LOST → pose validada | Inicio búsqueda → validada | Frame de máscara → validación |
|---|---:|---:|---:|
| EfficientSAM3 EV-M, sin ventana | 1.695 s | 1.243 s | 91 → 130 |
| EfficientSAM3 EV-M, con ventana | 1.910 s | 1.464 s | 92 → 138 |
| SAM3, sin ventana | 1.980 s | 1.526 s | 92 → 137 |

Tiempos calculados de `wall_s` de publicación; incluyen detección de pérdidas,
esperas y validación según columna. No constituyen un A/B controlado de modelos.
El tracking observado sigue aproximadamente en 4–5 actualizaciones/s; la captura
y el bombeo de ventana son independientes. Se inspeccionó la imagen guardada de
la pose 20 del ensayo GUI: el CAD vuelve a cubrir el ratón.

Resultados y logs: `auto_recovery_injected_20261011`,
`auto_recovery_gui_20261011`, `auto_recovery_sam3_20261011` (JSONL y log en esta
carpeta). El segundo incluye imágenes en `auto_recovery_gui_20261011_frames`.

15 pruebas unitarias aprobadas: recuperación (single-flight, transitorios,
backoff, errores, invalidación por recuperación espontánea y reinicio manual,
rechazo de profundidad inválida), SAM, cliente EfficientSAM y display.
Compilación Python y `git diff --check` correctos.

## SAM 3.1: revisión documental, sin benchmark local

Se revisaron la [ficha oficial](https://huggingface.co/facebook/sam3.1) y las
[notas oficiales de lanzamiento](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).
La versión introduce Object Multiplex: comparte memoria y procesa objetos en
grupos para acelerar seguimiento de múltiples máscaras en video. Meta informa
~7× a 128 objetos en una H100 y mejoras VOS en seis de siete benchmarks; los
resultados de segmentación conceptual de video son mixtos.

La ficha indica que estos checkpoints no tienen integración Transformers.
Nuestro cargador actual usa `Sam3Model` de Transformers; necesita otro backend
con el código oficial, no un simple cambio del nombre del checkpoint.

Es candidato a evaluar para continuidad de máscaras, pero esos resultados no
demuestran una mejora para un solo ratón en nuestra RTX 4070 Laptop. No se
instaló ni midió SAM 3.1 en esta tarea, y no reemplaza el cálculo de profundidad
ni de pose 6D de FoundationPose. La recuperación implementada funciona con los
dos backends ya probados: SAM3 y EfficientSAM3.

## Cámara

```powershell
uv run --project pose python pose/live_pose.py --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --scale 0.5 --valid-iters 4 --track-iters 5 --mask-text 'computer mouse' --fast-init --sam-backend efficientsam3
```

Sin `--sam-backend efficientsam3` usa SAM3. Para repetir la prueba controlada,
cambiar el script a `pose/validate_live_recovery.py` y añadir `--bag
data/bags/live_20261010_030134.db3 --frames 30 --metrics-jsonl <archivo_nuevo>`.
