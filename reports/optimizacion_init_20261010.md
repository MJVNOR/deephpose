# Inicialización: 10 de octubre de 2026

El registro evalúa 252 orientaciones, las refina cinco veces y selecciona la mejor.
El perfil aislado caliente del original dio 3.34 s: 2.63 s refinando y 0.68 s
puntuando. Durante la inicialización de cámara, FoundationStereo seguía usando
la GPU para generar frames que la inicialización descartaba.

## Cambios activos por defecto

- Pausa de nuevas inferencias FS mientras se ejecuta SAM o el registro. Una
  petición ya enviada termina; la captura conserva el último frame disponible.
- Reutilización de geometría de recortes en refinamiento y puntuación. Se
  mantienen candidatos, iteraciones, interpolación y vaciados de caché CUDA.
- El HUD separa preparando, refinando 252 y seleccionando 252. Los métodos
  originales se restauran incluso cuando falla el registro.
- Cálculo del diámetro de la malla en bloques de 128 filas, conservando el
  muestreo y las mismas operaciones NumPy. La prueba completa detectó un fallo
  de asignación de RAM en la matriz original de 10,000 × 10,000 distancias.
  Para 10,000 puntos float64, el temporal de diferencias de tres coordenadas
  pasa de 2.24 GiB a 29.3 MiB. Esta corrección de carga se aplica también al
  modo de referencia; no forma parte de los tiempos de registro de la tabla.

El comando habitual no cambia; `--reference-init` recupera el comportamiento
anterior para comparar. No se modificaron los archivos de FoundationPose.

## Mediciones

RTX 4070 Laptop, escala 0.5, FS4, registro de cinco iteraciones. A/B en orden
original, optimizado, optimizado, original, con los mismos datos y máscara.

| Registro con FS concurrente | Original | Optimizado |
|---|---:|---:|
| Primera muestra | 14.97 s | 7.22 s |
| Segunda muestra | 23.60 s | 9.68 s |
| Media | 19.29 s | 8.45 s |

Reducción observada de 56.2%. El ensayo repite IR grabado en un proceso FS;
simula la competencia de GPU, pero no es una medición de cámara física. Hay solo
dos muestras por variante y variabilidad considerable: no garantiza esos tiempos
en cada arranque. Excluye carga e inferencia de SAM y carga de modelos.

Sin FS concurrente, los registros calientes originales tardaron 3.39–3.44 s y
los optimizados 3.33–3.45 s. La reutilización de geometría por sí sola aporta
poco aquí; la mejora principal procede de pausar trabajo concurrente.

## Equivalencia y validación

Tres frames (60, 1260 y 2460), cuatro registros por frame: igualdad exacta de
la pose seleccionada, las 252 poses candidatas ordenadas y sus puntuaciones.
En las cuatro pruebas concurrentes también coincidió exactamente la pose final.
El primer frame usa máscara SAM guardada; los otros usan proyecciones CAD de
poses guardadas. Esto verifica equivalencia, no exactitud absoluta ni nuevas escenas.

La prueba unitaria cubre progreso y restauración de métodos tanto con éxito
como con excepción, en ambas variantes. Otra comprueba igualdad exacta del
diámetro y del estado del generador aleatorio con float32/float64, con y sin
muestreo, cruzando varios bloques. Las cuatro pruebas de geometría/tracking
también pasan.

Evidencia: [perfil](init_profile_20261010/summary.json),
[equivalencia y tiempos](init_equivalence_20261010/summary.json).

## Prueba completa con SAM y ventana

La reproducción de `live_20261010_030134.db3` completó SAM → INIT → cuatro
poses TRACK y cerró normalmente. SAM obtuvo 14,466 píxeles con score 0.98;
no hubo LOST, con fracción válida aproximada de 0.942 y residuo 2.1–2.2 mm.
Se usaron escala 0.5, FS4 y tracking5, sin máscara ni pose precargadas.

En esta pasada el registro tardó **18.02 s**, por encima del ensayo A/B.
Durante la ejecución Windows reportó solo unos 580 MiB libres de memoria
virtual y 3 GiB de RAM física libre; `vmmemWSL` tenía 31.79 GiB de memoria
privada y había otros procesos Python. No se cerraron procesos ajenos a la
prueba. Es evidencia de presión de memoria, no una medición causal aislada
de su impacto. También FS y tracking fueron más lentos que en los ensayos
previos. No usar esta pasada para prometer tiempos ni extrapolar FPS.

La primera pasada falló cargando la malla por una asignación de 763 MiB;
la segunda, con diámetro calculado por bloques, completó todo el flujo.
Evidencia: [telemetría](init_live_smoke_v2_20261010.jsonl),
[log completo](init_live_smoke_v2_20261010.log),
[fallo previo de memoria](init_live_smoke_20261010.log).
