# deephpose — Plan pose 6D mouse (FoundationPose + FoundationStereo)

Entorno: Windows 11 nativo, Python 3.11, uv, RTX 4070 Laptop 8 GB, RealSense D415. Sin WSL2/Conda/Docker. No reconstruir app: reutilizar `scripts/` actuales.
Orden: 1 geometría → 2/3 profundidad-GPU → 4 FoundationPose Windows → 5 malla → 6 estática → 7 vivo.

## 1. Geometría `scripts/project_to_rgb.py` — hecho
- [x] Z-buffer min-Z explícito + índice ganador (`splat_to_rgb`: unique sobre píxel lineal, primera aparición en orden asc = mínimo).
- [x] Rotación `reshape(3,3,order="F")` column-major, verificado vs `rs2_transform_point_to_point` (err col ~1e-7, row ~5-10 mm).
- [x] Distorsión RGB: `rectify_color` + `model` guardado en `capture.py:intr_to_dict`; salida = RGB rectificado + `K_rect` (`K_rect.json` + `color_rect.png`). Capturas viejas sin `model` → aviso + asunción explícita.
- [x] `project_ir_to_rgb()` reutilizable estático/vivo. Retorna `Z_rgb (f32, inf), mask_valid, K_rect`.
- [x] `.npy` intacto; colormap/inpaint solo `overlay.png`. Vacío → `inf`, `0%`, `zmed=sin-puntos`.
- [x] Tests `tests/test_project.py` (asserts planos, sin pytest): cercano gana; match SDK; escala/geometría + vacío.
- Verif: `uv run python tests/test_project.py` OK; real `emitter-on` 286166/288996 93.2% zmed 0.86 m, `emitter-off` 285836/289141 93.0% zmed 0.87 m.

## 2. Profundidad y captura `disparity_to_depth.py + capture.py + live.py` — hecho (pendiente smoke vivo por USB)
- [x] Fórmula única: `disp_to_depth()` pura + `check_stereo_cal()`; `live.py` la usa (borrada `fxB` `live.py:143`).
- [x] Validación: IR rectificadas (coeffs≈0), `ppx1≈ppx2` (1px), escala efectiva real `disp/cal` (aviso si ≠ pedida), `z_min/z_max` de `config.toml` (`--z-min/--z-max`), NaN/+inf/`d≤0` → `inf`, baseline inválido aborta.
- [x] Esquema unificado: `live.py` usa `capture.intr_to_dict` (con `model`) + `depth_intrinsics` + aviso viewpoint; metadata en dicts `timestamps_ms/frame_numbers` + `ts_domain/dt_ms/emitter_actual/warmup` (lectura de capturas viejas intacta: JSON).
- [x] Vivo: `dt_max_ms` (`--dt-max`, descarta desincronizados), `ts_domain` registrado, `.copy()` antes de cola, `warmup` config (`--warmup`), `started/th/stop` init antes de `try` + `join` + `pipe.stop()` guardado, `OutOfMemoryError` distinguido de `RuntimeError`.
- [x] Checks `tests/test_depth.py`: no-finitos/rango, abortos calibración, escala efectiva + compat sin `model`.
- Verif: `test_depth.py` OK, `test_project.py` OK, `project_to_rgb` real idéntico (286166/288996 93.2% 0.86 m), `live.py --help` OK.
- [x] Smoke `live.py --frames 3` OK tras cable USB 3.2 (`usb_type_descriptor=3.2`): 3 frames, cierre limpio, cámara liberada.

## 3. GPU `fs_model.py` + color — hecho
- [x] `infer_disp` con CUDA events + `synchronize`: `t_h2d/t_infer/t_d2h/t_total` separados; picos `alloc` (pesos+activaciones) y `reserved` (+caché allocator, no otros procesos).
- [x] `infer.py` guarda claves viejas (`t_infer_s`, `vram_peak_mib` → `evaluate.py` intacto) + nuevas; `live.py` usa `info["t_infer_s"]`.
- [x] `turbo_norm` devuelve BGR directo (antes RGB a `imshow` → colores swapped); `evaluate/project` ya eran consistentes.
- Verif: `infer` real `t_infer=0.65s t_total=0.66s alloc=255MiB reserved=2104MiB pico_alloc=1603MiB` (coherente; `t_infer` en línea con 0.62-0.66 históricos); `evaluate` MAE=11.5mm idéntico; `live --frames 3` OK; tests OK.

## 4. FoundationPose Windows (ref: https://github.com/NVlabs/FoundationPose) — hecho
- [x] Toolchain sistema: CUDA Toolkit 12.4.1 (`nvcc V12.4.131`), `CUDA_HOME` usuario fijado, cmake 4.4.4, MSVC 14.44. Sin tocar `torch/torchvision/xformers/flash-attn`; `uv.lock` intacto.
- [x] Auditados imports model-based: compilar pytorch3d + nvdiffrast + mycpp; pip `kornia/ruamel.yaml/pandas/transformations/omegaconf/h5py/sklearn/warp-lang==1.8.0`; omitidos kaolin/pyrender/pyOpenGL/mycuda (no están en la cadena del demo).
- [x] Proyecto uv separado `pose/` con `torch 2.6.0+cu124`; vcpkg Boost (system, program-options, assign, format, algorithm) + Eigen3.
- [x] Demo oficial `run_demo.py` completa: 737 poses `debug/ob_in_cam/` (register + track). Detalle en `pose/WINDOWS.md`.
- Nota: `track_refine_iter` default upstream = 2 (paper cita 1); editable, se fija en paso 6.

## 5. Malla `scripts/prepare_mesh.py` — hecho (pendiente confirmación física tuya)
- [x] Localizado `3D Models/*.obj+.mtl+images/` (8 materiales PBR, texturas completas).
- [x] `prepare_mesh.py`: reescribe solo vértices `v'=s·v+t` (caras/UV/MTL intactos), `--units` declarado, `--origin keep|center`, `--expect L,W,H` verifica tamaño físico, guarda `transform_original_to_prepared.json`, reverifica recargando.
- [x] Medido: 8 partes, 79860 caras, 59062 verts, ext [88.3, 60.3, 122.3] mm (ejes x,y,z), UVs OK, 0 texturas perdidas. Carga en env `pose/` con normales.
- [ ] TUYO: confirma que [88.3, 60.3, 122.3] mm encaja con tu mouse físico (largo≈122, ancho≈88, alto≈60) o pasa `--expect` medido; si el alto no cuadra, revisamos orientación antes del paso 6.

## 6. Pose estática model-based `pose_model.py + pose.py` — hecho V1 (1 captura)
- [x] `pose/pose_model.py`: multitela→vertex-colors, `estimate_pose` (coarse+refine5+scorer), `track_pose` (solo refine), `render_depth`/`fit_metrics`/`draw_overlay` (rojo solo display, scores=ranking).
- [x] `pose/pose_static.py` CLI reusa `infer/project`; `depth inf→0` en borde API, máscara original intacta.
- [x] Primera pose real `20261008_040430` (máscara gruesa bbox+profundidad, documentada): `valid_frac=0.94`, `resid_med=0.8mm`, `t_est=11.4s`, `vram=138MiB`. Salidas en `data/processed/<cap>_pose/` + `meta_pose.json`.
- [x] Bugs de frames cazados: tensores CUDA→numpy, orden retornos render, malla centrada vs `best_pose` (silueta con `pose_last`), ejes al centro (origen OBJ fuera de frame).
- [x] Segunda captura `20261008_044401` (ratón girado): `valid_frac=0.944`, `resid_med=1.2mm`, CAD encaja visualmente; rotación relativa 61.6° coherente con el giro físico. Validación cruzada OK.
- [x] Serie estabilidad (3 capturas, ratón quieto): frac 0.940±0.000, residuo ≤1.5mm; jitter entre pares 0.5-3.1° / 1.8-7.5mm (t std [1.7,0.2,2.7]mm). Repetible para V1; el vivo deberá suavizar.

## 7. Validar → vivo `live_pose.py` — en curso
- [x] Track por etapas `pose/track_burst.py` (5 frames con movimiento): INIT 15.9s, TRACK 0.14-0.21s, frac≈0.94, residuo ~1mm, 0 LOST. VRAM track 190MiB.
- [x] Hallazgo: `Utils` top-level colisiona entre repos → worker FS en proceso aparte (`scripts/fs_worker.py`, protocolo líneas) + `pose/live_pose.py` orquesta (captura porteada, proyección, register/track, HUD lat/edad/fps/estado, r/s/q).
- [ ] Smoke `live_pose --frames 3` + medición conjunta 8GB + serie viva con movimiento.

## 8. Config y entregables
- [ ] `config.toml:[pose] mesh, scale, pesos, mask, iters_init/track, umbrales lost, viz_alpha`. CLI gana.
- [ ] README `uv run` reproducibles: `prepare_mesh/pose/live_pose`. No subir pesos/capturas/mallas. Registrar revs externas.
- [ ] Reportar cambios, verificaciones, VRAM (`t_fs/t_init/t_track` separados), bloqueos. No vender cifras README como precisión absoluta.
