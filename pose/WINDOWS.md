# FoundationPose en Windows 11 nativo (`pose/`)

Env separado: mismo `torch 2.6.0+cu124` que deephpose, `uv.lock` principal intacto.
Puente con el proyecto principal vía ficheros (`data/raw`, `data/processed`).

## Toolchain (todo verificado el 2026-10-08)

- CUDA Toolkit 12.4.1 (`nvcc V12.4.131`), `CUDA_HOME` a nivel usuario.
- MSVC 14.44 (VS2022 Community) + cmake 4.4.4 + vcpkg.
- Compilado desde fuente: `pytorch3d==0.7.9` (~10 min), `nvdiffrast==0.4.0` (~2 min),
  `mycpp` (cmake+Ninja, toolchain vcpkg).
- vcpkg: `boost-system, boost-program-options, boost-assign, boost-format,
  boost-algorithm, eigen3` (x64-windows).
- pip: `kornia, ruamel.yaml, pandas, transformations, omegaconf, h5py,
  scikit-learn`, **`warp-lang==1.8.0` (fijado: 1.18 exige driver CUDA ≥13.0)**.
- Omitidos (ruta model-based no los importa): `kaolin, pyrender, mycuda`.

## Adaptaciones Windows (documentadas, upstream intacto salvo 1)

1. `FoundationPose/mycpp/include/Utils.h`: `#include <unistd.h>` tras guard
   `_WIN32` (nada de `src/` usa símbolos POSIX — verificado por grep).
2. Configure `mycpp`: `-DPYTHON_EXECUTABLE=<pose/.venv python>` (pybind11 usa el
   módulo antiguo y pescaba un shim 3.10 del PATH) + `-DCMAKE_CXX_FLAGS=
   /D_USE_MATH_DEFINES` (por `M_PI` en `pybind_api.cpp`).
3. `run_demo.py:33` trae `os.system('rm -rf …')` (no existe en Windows, falla
   silencioso): precrear `debug/track_vis debug/ob_in_cam` antes de correr.

## Comprobaciones

- `uv run --project pose python -c "import estimater"` → OK (torch, pytorch3d,
  nvdiffrast, mycpp, warp, kornia).
- `python run_demo.py` (mostaza, `est_refine_iter=5`) → 737 poses en
  `debug/ob_in_cam/` (frames 0–736: register + track). Nota: `track_vis/` solo
  con `--debug 2`.
- `run_demo.py` trae `track_refine_iter=2` por defecto (el paper cita 5/1);
  es flag editable, se fijará en `config.toml:[pose]`.
