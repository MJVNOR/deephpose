# deephpose — FoundationStereo + RealSense D415 (rumbo a estimación de pose)

Profundidad estéreo con FoundationStereo (ViT-small `11-33-40`) sobre pares IR de
una D415, comparada con la profundidad original de la cámara y proyectada a RGB.
Windows 11 nativo, RTX 4070 Laptop 8 GB, Python 3.11, `uv`. Sin WSL/Conda/Docker.

## Instalación

```powershell
uv python pin 3.11
uv sync   # torch 2.6.0+cu124 desde el índice oficial (ver pyproject.toml)
```

Terceros: `flash-attn` no publica wheel Windows; se usa
`kingbri1/flash-attention` v2.8.3 (`cu124+torch2.6.0+cp311`, fijado en
`uv.lock`). Los parches offline del issue #219 (dinov2 local, edgenext bin) no
hicieron falta: con red, timm/dinov2 bajan solos de HuggingFace.

Pesos: `pretrained_models/11-33-40/` (`cfg.yaml` + `model_best_bp2.pth`, 787 MB,
Drive oficial). FoundationStereo está clonado sin modificar en
`third_party/` (`6e88068`). Solo se adapta `torch.load(weights_only=False)` en
nuestro wrapper: torch 2.6 cambió el default y el checkpoint es de 2025.

## Uso

```powershell
uv run python scripts/capture.py --emitter on      # data/raw/<sello>_emitter-on/
uv run python scripts/infer.py --capture data/raw/<sello>_emitter-on
uv run python scripts/disparity_to_depth.py --disp <..._fs/disp.npy> --calibration <.../calibration.json> --out <dir>
uv run python scripts/project_to_rgb.py --fs <..._fs> --capture <...>
uv run python scripts/evaluate.py --fs <..._fs> --capture <...>
uv run python scripts/live.py --emitter on          # s=guardar, q=salir
uv run python scripts/live.py --view both           # compara D415 | FS
```

`config.toml` centraliza resolución, emisor, checkpoint, iters y rango; el CLI
gana. Demo oficial (sin tocar `third_party`):
`uv run python scripts/run_official_demo.py --scale 1`.

## Resultados medidos (4070 libre, 640×480, valid_iters=16)

`t_load≈2.6 s`, `t_infer≈0.64 s`, `vram≈1646 MiB`, AMP válida.
Misma escena: emisor ON → MAE 11.5 mm / válido 86.6%; OFF → MAE 51.7 mm /
44.8%. El emisor aporta cobertura D415, no precisión FS. Figuras en
`reports/figures/`, números en `data/processed/*_eval/metrics.json`.

## Limitaciones

- 8 GB obligan a VRAM libre: otro proceso con GPU mató la corrida a escala 1.0
  (sin traceback; con VRAM libre tarda 9.4 s). Si falta memoria: `--scale 0.5`.
- Sin Triton en Windows (aviso xformers, benigno); no se instaló por innecesario.
- D415 = comparación, no ground truth. Falta exactitud absoluta con distancia
  conocida y repetibilidad en escena fija controlada.
- Errores claros: sin cámara / cámara ocupada / streams no soportados /
  captura incompleta o desincronizada / pesos ausentes / sin GPU.
