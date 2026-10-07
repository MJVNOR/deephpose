# deephpose — Plan de trabajo

Equipo: Windows 11 nativo, RTX 4070 Laptop 8 GB VRAM, 64 GB RAM, RealSense D415 por USB 3.0, Python 3.11 64-bit, uv. Sin WSL2/Conda/Docker. Editor terminal: vim (no nano).

Repo oficial: https://github.com/NVlabs/FoundationStereo
Checkpoint inicial: ViT-small `11-33-40`.

Herramientas obligatorias: `obscura` (ver páginas web) y `marimo`, ambos **solo vía sus respectivos MCPs** con `execute` (nada de webfetch/curl para lo que cubra obscura; nada de editar `.py` de notebook en sesión viva, usar `cm`).

## 0. Inspección y entorno — hecho 2026-10-06

- [x] `.venv` con Python 3.11.15 (`uv venv --python 3.11`).
- [x] `uv python pin 3.11`, `pyproject.toml` + `uv.lock` creados, `uv run python` → 3.11.15 MSC 64-bit.
- [x] Archivos existentes: carpeta vacía salvo `.venv` + `plan.md`. Sin instalación global ni firmware tocados.
- [x] GPU: `nvidia-smi` → RTX 4070 Laptop GPU, driver 591.74, 8188 MiB. `torch.cuda` pendiente de Fase 1 (torch aún no instalado).
- [x] Vía MCP obscura leídos: README, `environment.yml` (python 3.11, torch 2.4.1/torchvision 0.19.1/xformers 0.0.28.post1, opencv-contrib-python), `scripts/run_demo.py` (args `--scale/--hiera/--valid_iters/--z_far/--get_pc`, `depth=K[0,0]*baseline/disp`), issue Windows #219 (torch 2.6.0+torchvision 0.21.0 cu124, xformers cu124, flash-attn wheel `kingbri1/flash-attention`, parches offline dinov2 local + edgenext_small `pytorch_model.bin`).
- [x] FoundationStereo `HEAD = 6e8806816b533e4d13ddbb95ffa907b797060a62`. Checkpoint inicial pendiente de descarga: ViT-small `11-33-40` (Drive oficial).

## 1. Dependencias Windows — hecho 2026-10-06

- [x] Índice oficial explícito en `pyproject.toml`: `[[tool.uv.index]] pytorch-cu124 = https://download.pytorch.org/whl/cu124` + `tool.uv.sources` para torch/torchvision/xformers.
- [x] Resolución conjunta: `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124`, `xformers 0.0.29.post3` (no copiadas de Linux 2.4.1/0.19.1/0.0.28, sino receta Windows issue #219).
- [x] `uv run python`: `cuda: True`, `NVIDIA GeForce RTX 4070 Laptop GPU`.
- [x] flash-attn: wheel de terceros `kingbri1/flash-attention` v2.8.3, asset exacto `cu124+torch2.6.0+cp311-win_amd64` (56 MB), `import flash_attn` → 2.8.3 OK. Sin adaptaciones de atención (código FS no importa `flash` directamente en core/dinov2).
- [x] OpenCV único: `opencv-contrib-python 5.0.0.93`.
- [x] Pesos `11-33-40` en `pretrained_models/11-33-40/` (`cfg.yaml` vit_size `vits` + `model_best_bp2.pth` 787 MB, cabecera ZIP válida; unpickle completo pendiente de `omegaconf`, va con el resto del env en Fase 2).
- [x] Fase 2 cerrada 2026-10-07: FS clonado en `third_party/` (`6e88068` = HEAD registrado), deps demo añadidas (timm, omegaconf, imageio, open3d, trimesh, joblib, pandas, hf-hub, pyyaml, scipy). Parches offline #219 NO necesarios (hay red: timm/dinov2 bajan de HF). Shim propio `scripts/run_official_demo.py` (torch 2.6 `weights_only=False` + `--no-viz`; third_party intacto).
- [x] Demo oficial OK con `11-33-40`, `valid_iters=16`, `--scale 0.5`: `data/processed/demo_oficial/` (depth 270×480 f32, mediana 0.49 m) + `reports/figures/demo_oficial_11-33-40.png`.
- [x] Veredicto escala 1.0 (2026-10-07, VRAM libre): **era contención, no límite del modelo** — 960×540 completo en 9.4 s pared (carga+infer+ply), depth 540×960 f32, 90.7% válidos>0, mediana 0.51 m, en `data/processed/demo_oficial_scale1/` + `reports/figures/demo_oficial_11-33-40_scale1.png`. Lección: medir siempre con VRAM libre; revalidar picos en `infer.py`.

## 2. Captura D415 (`scripts/capture.py`) — hecho 2026-10-07

- [x] IR1+IR2+depth 640×480@30fps verificada en cámara real (D415 `151322068842`).
- [x] Mismo frameset, `dt=0.0 ms`, warmup 60 framesets, `pipeline.stop()` en `finally`, errores claros (sin cámara / combo no soportada / incompleta / desincronizada).
- [x] Dos capturas en `data/raw/`: `*_emitter-on` y `*_emitter-off`, cada una con `left.png/right.png/depth_original.npy (480×640)/calibration.json/metadata.json`.
- [x] Calibración real: `fx=592.88` px, `|B|=0.055` m (signo según convención ir1→ir2), depth scale del sensor. Sin valores manuales.

## 3. Integración FoundationStereo (`scripts/infer.py`) — hecho 2026-10-07

- [x] IR mono → 3 canales replicados sin tocar intensidades; `eval()`, sin gradientes, un par a la vez.
- [x] Ambas capturas a 640×480, `valid_iters=16`, AMP validada (`amp_ok=True`).
- [x] Medido en 4070 libre: `t_load=2.7 s`, `t_infer=0.62-0.66 s`, `vram=1646 MiB`. Salidas: `disp.npy` + `meta.json` en `data/processed/<captura>_fs/`.

## 4. Disparidad → profundidad (`disparity_to_depth.py`) — hecho 2026-10-07

- [x] IR rectificadas (coeffs=0), `cx` IR1==IR2 → `Z=fx·|B|/d` sin corrección. `fx` px, `|B|=0.055` m de calibración real, `d` px misma resolución (focal escalada con `scale`).
- [x] Máscara `d>0` + rango `z_min/z_max` (0.2–10 m). Nube en frame IR-izq, sin intrínsecos RGB.

## 5. RGB + proyección (`capture.py` color, `project_to_rgb.py`) — hecho 2026-10-07

- [x] Captura guarda `color.png` + intrínsecos RGB + extrínseco IR-izq→RGB + intrínsecos depth (verifica viewpoint depth==IR1).
- [x] Proyección con z-buffer a geometría RGB: `depth_on_rgb.npy` + `overlay.png` + `cloud_color.ply` (puntos en IR-izq coloreados).
- [x] Verificado on/off: ~288k puntos, 93.8% cae en RGB, zmed ~0.88 m. Figuras en `reports/figures/overlay_rgb_{on,off}.png`.

## 6. Comparar y evaluar (`scripts/evaluate.py`) — hecho 2026-10-07

- [x] Panel IR-L/IR-R/disparidad/FS/D415/diferencia, misma colormap y geometría (D415 remuestreada nearest solo si difiere).
- [x] Misma escena on/off: ON → MAE 11.5 mm, med 2.1 mm, válido conjunto 86.6%; OFF → MAE 51.7 mm, med 11.6 mm, válido 44.8%. Lectura: el emisor suma cobertura D415, no precisión FS; D415 es comparación, no GT.
- [x] Tiempos de `meta.json` (`t_load≈2.5 s`, `t_infer≈0.63 s`, `vram=1646 MiB`) + `t_eval_s` en `metrics.json`. Repetibilidad en escena fija controlada: pendiente.

- [ ] Figura: IR-L/R, disparidad FS, profundidad FS, profundidad D415, diferencia en válidos. Misma colormap y geometría/resolución antes de restar.
- [ ] Métricas: `t_load`, `t_infer` post-warmup, `t_total`, pico VRAM, % válidos, MAE/mediana vs D415, varianza en escena fija.
- [ ] D415 = comparación, no GT; precisión absoluta con distancias conocidas/referencia externa.

## 7. Validación práctica (pendiente)

- [ ] Plano + piezas (poca textura, bordes, reflejos). Trípode, emisor on/off, log luz/distancia/config.
- [ ] Distinguir cobertura vs precisión. FPS medidos en esta laptop, no del PDF ni otra GPU.
- [ ] Repetibilidad en escena fija controlada + distancia conocida (exactitud absoluta).

## 8. Captura continua (`scripts/live.py`, tras validar estático) — hecho 2026-10-07

- [x] Hilo de captura + inferencia en principal, `Queue(maxsize=2)` descartando viejos, calibración pegada a cada item.
- [x] Muestra D415 | FS + `t_infer`/fps. `s` guarda captura formato `capture.py`, `q`/ESC sale, `--frames N` para pruebas. `pipe.stop()` + ventanas siempre.
- [x] Núcleo compartido `scripts/fs_model.py` (`infer.py` adelgazado y reverificado: mismos números). Smoke test `--frames 3` OK, cámara liberada.

- [ ] Hilos separados captura/inferencia, `queue(maxsize=1-2)` descartando viejos, calibración pegada a cada frame.
- [ ] Mostrar ambas profundidades + tiempos, guardar capturas bajo tecla, salir liberando cámara.

## Entregables

`capture/infer/disparity_to_depth/evaluate/live` + `config.toml` (resolución, emisor, checkpoint, iters, rango), `pyproject.toml+uv.lock`, README con PowerShell+uv reproducible, `results/` ejemplo con tiempos/VRAM/limitaciones, errores claros (sin cámara/pesos/OOM).

Pendiente: notebook marimo `notebooks/03-compara.py`.
