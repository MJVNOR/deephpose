"""Demo oficial sin tocar third_party: adapta torch.load a torch>=2.6 y permite saltar el visor.

- torch 2.6 cambió el default de weights_only a True; el checkpoint 2025
  (pesos oficiales NVIDIA, fuente de confianza) requiere weights_only=False.
- --no-viz omite el Visualizer bloqueante; los .png/.npy/.ply se generan igual.
"""
import runpy
import sys

import torch

_orig_load = torch.load


def _load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _load

# ponytail: default local, evita ensuciar la raíz
if "--out_dir" not in sys.argv:
    sys.argv += ["--out_dir", "data/processed/demo_oficial"]

if "--no-viz" in sys.argv:
    sys.argv.remove("--no-viz")
    import open3d as o3d

    o3d.visualization.Visualizer.run = lambda self: None  # ponytail: solo demo, live.py gestiona su ventana

runpy.run_path("third_party/FoundationStereo/scripts/run_demo.py", run_name="__main__")
