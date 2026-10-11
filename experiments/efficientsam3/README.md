# EfficientSAM3 experimental

Instalado en su propio entorno Python 3.12 / PyTorch 2.7.1 CUDA 12.6. No cambia
el entorno Python 3.11 de FoundationPose. Se probaron EV-M y TV-M; el vivo ofrece
EV-M como opción experimental. No se instaló MobileSAM.

Código upstream: https://github.com/SimonZeng7108/efficientsam3,
commit `bd0936c788fed8d51fa799437f05abd97b401b06`. Pesos:
`Simon7108528/EfficientSAM3`, revisión
`85b05896928f974e308f889d7ccb2eefc069de98`, carpeta `efficientsam3_ft`.
El código está en `third_party/EfficientSAM3` y los pesos en
`data/models/EfficientSAM3`; ambos están ignorados por Git.

Para reconstruir en otra instalación, clonar el repo en esa ruta y seleccionar
ese commit. Luego, desde la raíz del proyecto:

```powershell
uv sync --project experiments/efficientsam3
uv run --project experiments/efficientsam3 python experiments/efficientsam3/download_models.py
```

`backend.py` usa la arquitectura B1 que corresponde al checkpoint EV publicado,
MobileCLIP-S0 y contexto 16. Carga todos los pesos con `strict=True`; una
incompatibilidad aborta en lugar de dejar capas sin cargar. Para TV usa TinyViT
11M. Se usa autocast BF16 y resolución 1008; los pesos permanecen en FP32.

## Cámara

```powershell
uv run --project pose python pose/live_pose.py --mesh data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj --scale 0.5 --valid-iters 4 --track-iters 5 --mask-text 'computer mouse' --fast-init --sam-backend efficientsam3
```

El umbral experimental de EV es 0.1. Se eligió explorando esta misma grabación:
no está calibrado con un conjunto independiente ni demuestra generalización.
`--sam-threshold` permite cambiarlo. El umbral no es una probabilidad calibrada.
Sin `--sam-backend efficientsam3` se conserva SAM3 con umbral 0.5.

El worker permanece residente y transmite RGB/máscara por el protocolo binario
del proyecto. La selección de la máscara sigue siendo el mayor score; si no hay
detección, el vivo reintenta. No reutiliza una máscara vieja ni cambia a SAM3
silenciosamente. Al salir se cierra el proceso de EfficientSAM3.

## Pruebas repetibles

Ejecutar en serie para no competir por GPU. Usar carpetas de salida nuevas.

```powershell
uv run --project pose python experiments/efficientsam3/benchmark_masks.py --backend sam3 --out reports/sam3_nuevo
uv run --project experiments/efficientsam3 python experiments/efficientsam3/benchmark_masks.py --backend efficient --threshold 0.1 --out reports/efficient_ev_nuevo
uv run --project experiments/efficientsam3 python experiments/efficientsam3/benchmark_masks.py --backend efficient --variant tv --threshold 0.1 --out reports/efficient_tv_nuevo
```

`diagnose.py` compara textos y BF16/FP32 en el frame inicial, sin aceptar sus
máscaras de score bajo como detecciones de producción. `compare_pose.py` usa
los resultados guardados del 11 de octubre para medir el efecto de la máscara
en FoundationPose con la misma configuración. `summarize.py` genera la tabla
y el mosaico visual de esos resultados.

[Informe completo](../../reports/efficientsam3_20261011.md).
