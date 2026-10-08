"""Prepara la malla del objeto para FoundationPose sin romper materiales.

Estrategia mínima: el OBJ multitela se conserva tal cual (caras, UV, MTL,
texturas); solo se reescriben las líneas `v x y z` con v' = s·v + t.
Guarda la transformación original→preparado en JSON. No adivina unidades:
el usuario declara `--units` y verifica dimensiones físicas con `--expect`.

Uso:
  uv run python scripts/prepare_mesh.py --in "3D Models/modelo.obj" --out data/mesh/mouse --units m --origin keep --expect 126,84.3,51
  (--expect: largo,ancho,alto medidos en mm, en el orden de ejes del OBJ)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="OBJ original (no se modifica)")
    p.add_argument("--out", required=True, help="directorio de salida")
    p.add_argument("--units", choices=list(UNIT_SCALE), default="m",
                   help="unidades declaradas del OBJ original")
    p.add_argument("--origin", choices=["keep", "center"], default="keep",
                   help="keep: solo escala; center: bbox al origen")
    p.add_argument("--expect", default=None,
                   help="extents físicos L,W,H en mm (orden de ejes del OBJ), p.ej. 126,84.3,51")
    p.add_argument("--tol", type=float, default=0.10, help="tolerancia relativa de --expect")
    return p.parse_args()


def main():
    a = parse_args()
    src = Path(a.inp)
    if not src.is_file():
        sys.exit(f"error: no existe {src} (pasa --in al OBJ descargado)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    scene = trimesh.load(src, process=True)
    geoms = dict(scene.geometry) if isinstance(scene, trimesh.Scene) else {"mesh": scene}
    if isinstance(scene, trimesh.Scene):
        world = scene.to_geometry()  # solo para medir el bbox global
    else:
        world = scene
    lo, hi = np.asarray(world.bounds, float)
    ext_orig = hi - lo
    s = UNIT_SCALE[a.units]
    ext_m = ext_orig * s
    if not (0.01 <= ext_m.max() <= 2.0):
        sys.exit(f"error: extents {ext_m} m fuera de rango objeto de mesa; revisa --units")
    if a.expect:
        exp = np.array([float(x) for x in a.expect.split(",")]) / 1000.0
        if exp.shape != (3,):
            sys.exit("error: --expect necesita 3 valores L,W,H en mm")
        rel = np.abs(np.sort(ext_m) - np.sort(exp)) / np.maximum(np.sort(exp), 1e-9)
        # ponytail: comparación ordenada = sólo verifica tamaño, no qué eje es cada dimensión
        if (rel > a.tol).any():
            sys.exit(f"error: extents {np.sort(ext_m)*1000} mm ≠ --expect (tol {a.tol}); "
                     "mide el mouse físico o corrige --units")
    t = -s * (lo + hi) / 2 if a.origin == "center" else np.zeros(3)

    n_faces = sum(len(g.faces) for g in geoms.values())
    missing_tex, no_uv = [], [n for n, g in geoms.items()
                              if getattr(getattr(g, "visual", None), "uv", None) is None]
    mtl = None
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("mtllib "):
            mtl = src.parent / line.split(None, 1)[1].strip()
        if line.startswith("map_"):
            tex = src.parent / line.split(None, 1)[1].strip().split()[-1]
            if not tex.is_file():
                missing_tex.append(tex.name)
    if mtl and not mtl.is_file():
        sys.exit(f"error: MTL referenciado no existe: {mtl}")
    print(f"partes={len(geoms)} caras={n_faces} "
          f"ext_orig={np.round(ext_orig, 4).tolist()} ({a.units}) "
          f"ext_m={np.round(ext_m*1000, 1).tolist()} mm uv_falta={no_uv} tex_falta={missing_tex}")

    # Reescritura solo de vértices; resto del fichero byte-idéntico.
    dst_obj = out / (src.stem + "_prepared.obj")
    nv = 0
    with open(src, encoding="utf-8", errors="replace") as fi, open(dst_obj, "w", encoding="utf-8") as fo:
        for line in fi:
            if line.startswith("v "):
                p = np.fromstring(line[2:], float, sep=" ")
                fo.write(f"v {p[0]*s+t[0]:.6f} {p[1]*s+t[1]:.6f} {p[2]*s+t[2]:.6f}\n")
                nv += 1
            else:
                fo.write(line if line.endswith("\n") else line + "\n")
    if mtl:
        shutil.copy2(mtl, out / mtl.name)
        for sub in ("images", "textures", "maps"):
            if (src.parent / sub).is_dir():
                shutil.copytree(src.parent / sub, out / sub, dirs_exist_ok=True)

    # Verificación recargando lo preparado (no fiarse del texto).
    chk = trimesh.load(dst_obj, process=True)
    chk_w = chk.to_geometry() if isinstance(chk, trimesh.Scene) else chk
    ext_chk = np.diff(np.asarray(chk_w.bounds, float), axis=0)[0]
    n_chk = sum(len(g.faces) for g in (dict(chk.geometry) if isinstance(chk, trimesh.Scene) else {"m": chk}).values())
    assert n_chk == n_faces, f"caras {n_chk} ≠ {n_faces}"
    assert np.allclose(np.sort(ext_chk), np.sort(ext_m), rtol=1e-4), (ext_chk, ext_m)
    (out / "transform_original_to_prepared.json").write_text(json.dumps({
        "src": src.name, "units": a.units, "scale": s, "translation_m": t.tolist(),
        "origin": a.origin, "ext_orig": ext_orig.tolist(), "ext_prepared_m": ext_chk.tolist(),
        "faces": n_faces, "vertices": nv, "parts": sorted(geoms),
        "uv_missing_parts": no_uv, "tex_missing": missing_tex,
    }, indent=2))
    print(f"OK {dst_obj} caras={n_chk} ext_m={np.round(ext_chk*1000, 1).tolist()} mm")


if __name__ == "__main__":
    main()
