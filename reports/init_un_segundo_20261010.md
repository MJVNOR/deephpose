# SAM + registro próximos a un segundo

El objetivo se mide desde un frame RGB-D disponible hasta la pose, con modelos
preparados. No se ha conseguido iniciar el proceso completo desde cero en un
segundo. Cargar SAM en la prueba aislada tardó 17.75 s; su primera segmentación,
1.20 s, y las siguientes 0.36–0.46 s. No se cambió el modelo ni su resolución.

## Implementación experimental

`--fast-init` precarga y conserva SAM, prepara kernels de pose y reduce la
búsqueda de 252 a 42 orientaciones repartidas en el espacio de rotaciones.
Refina una vez todas las orientaciones y cuatro veces las ocho mejor puntuadas.
Hay una nueva puntuación final. El registro habitual sigue disponible sin la
opción. La pose puede cambiar; esta optimización no tiene equivalencia numérica
con la búsqueda completa.

El calentamiento usa una entrada sintética y restaura el estado anterior; no
publica ni reutiliza esa pose. Cada inicialización real ejecuta SAM sobre su
propio frame. El bucle consulta las tareas pendientes con menor espera, evitando
añadir hasta 200 ms entre fases cuando FS está pausado.

La comprobación de profundidad y máscara puede activar la búsqueda completa:
cobertura válida menor de 0.3, residuo mediano mayor de 20 mm o IoU de máscara
menor de 0.5. Es un filtro de fallos gruesos, no una garantía de exactitud.
El fallback, una falta de detección o presión de memoria rompen el objetivo de
un segundo. Mantener SAM residente aumenta el consumo de memoria.

## Primeras mediciones

Con máscaras fijas de tres frames, el registro reducido tardó 0.50–0.92 s
frente a varios segundos del completo. Las diferencias de pose frente a la
búsqueda completa quedaron hasta 1.11° y 1.08 mm en ese ensayo particular.

La primera prueba conjunta cargó SAM, FoundationPose y un worker FS, pausado
durante la inicialización. En cuatro frames y dos repeticiones, SAM + registro
tardaron 0.84–0.96 s después del primer uso de los kernels de registro; ese
primer uso tardó 1.67 s. Las diferencias llegaron a 1.17° y 4.40 mm.
El frame 1740 no produjo máscara SAM: la primera versión del benchmark se
interrumpió ahí. El benchmark corregido registra esa falta de detección y sigue
con los demás frames; no la contabiliza como inicialización exitosa.

La diferencia entre ensayos refleja, entre otras cosas, que la creación del
estimador original muestrea aleatoriamente puntos para su diámetro; las
comparaciones rápidas/completas de cada ensayo comparten el mismo estimador.
No son errores frente a ground truth. Todas las pruebas usan la grabación
original del ratón, no una validación universal de otras orientaciones o escenas.

## Reproducibilidad

La prueba final con calentamiento previo completó cinco frames detectados
(60, 420, 780, 1260 y 2460), dos veces cada uno:

| SAM + registro, con modelos listos | Tiempo |
|---|---:|
| Media de 10 inicializaciones | 0.950 s |
| Mínimo | 0.894 s |
| Máximo | 1.064 s |

No se activó el fallback en esas diez inicializaciones. El frame 1740 volvió
a no producir máscara: cinco de seis frames intentados tuvieron detección.
No se incluye el intento sin detección en la media de inicializaciones exitosas.
La preparación de SAM tardó 13.97 s y la de los kernels de pose 1.15 s, aparte
de cargar los otros modelos y preparar profundidad.

En este ensayo las diferencias frente al registro completo llegaron a 0.80° y
2.92 mm. Los 20 pasos posteriores de tracking de cuatro segmentos no cumplieron
los criterios de LOST. El último frame no tiene frames posteriores en la muestra.
El resultado es aproximadamente un segundo en estas condiciones, no un límite
duro de latencia ni una garantía de precisión. Las variantes con SAM residente
tienen menos memoria libre para una búsqueda completa si necesitan fallback.

- [Perfil de SAM](sam_one_second_profile_20261010/summary.json).
- [Búsqueda reducida frente a completa](init_shortlist_20261010/summary.json).
- [Primera prueba conjunta, parcial](fast_startup_20261010/summary.json).
- [Prueba conjunta con calentamiento previo](fast_startup_v2_20261010/summary.json).

Los scripts están documentados en `pose/REPLAY.md`. Las pruebas unitarias
verifican selección de orientaciones sin duplicados, presupuesto 1 + 4 de
iteraciones, restauración incluso ante excepciones, umbrales de fallback y
tratamiento de detección vacía/error en SAM.

## Validación con ventana

El comando de cámara con `--fast-init`, añadiendo la grabación original para
reproducirla, completó 20 poses (INIT y 19 TRACK), sin LOST. SAM tardó 0.437 s,
el registro 0.518 s y el total incluyendo coordinación de hilos 0.979 s.
No hubo fallback. Las capturas inicial y final muestran la superposición sobre
el ratón, con residuo de profundidad de aproximadamente 2 mm.

La preparación de SAM fue de 12.25 s y la de pose de 1.17 s. La primera pose
apareció aproximadamente 46 s después de iniciar el script, incluyendo imports,
carga de modelos y primer cálculo de profundidad. El tiempo del frame hasta
publicar la primera pose fue 7.81 s, de los que 6.72 s correspondieron a la
primera petición al worker FS. **El arranque completo desde cero no tarda un
segundo**; el objetivo comprobado es la fase SAM + registro con modelos listos.

Evidencia: [telemetría](fast_init_live_20261010.jsonl),
[log](fast_init_live_20261010.log),
[primer resultado](fast_init_live_20261010_frames/00001_frame_000058.jpg),
[seguimiento](fast_init_live_20261010_frames/00020_frame_000238.jpg).
