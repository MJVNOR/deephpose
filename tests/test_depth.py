"""Checks del paso 2: fórmula única, validación y escala efectiva. `uv run python tests/test_depth.py`."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from disparity_to_depth import check_stereo_cal, disp_to_depth, load_capture_depth  # noqa: E402


def _cal(**kw):
    base = {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0,
            "width": 640, "height": 480, "coeffs": [0, 0, 0, 0, 0]}
    ir = dict(base, **kw)
    return {"ir1": dict(ir), "ir2": dict(ir), "baseline_m": -0.055}


def test_rechaza_no_finitos_y_rango():
    fx, B = 600.0, 0.055
    disp = np.array([[10.0, 0.0, -3.0, np.nan, np.inf]], dtype=np.float32)
    z, valid = disp_to_depth(disp, fx, B, 0.2, 10.0)
    assert valid.tolist() == [[True, False, False, False, False]], valid
    assert np.isinf(z[0, 1:]).all(), "inválidos van a inf, no a 0 ni NaN"
    assert abs(z[0, 0] - fx * B / 10.0) < 1e-6
    z2, v2 = disp_to_depth(np.full((2, 2), 1000.0), fx, B, 0.2, 10.0)  # ~3cm < z_min
    assert not v2.any() and np.isinf(z2).all(), "z_min se aplica"


def test_calibracion_incompatible_aborta():
    for bad in (dict(_cal()["ir2"], ppx=400.0),):  # ppx difieren
        try:
            check_stereo_cal({"ir1": _cal()["ir1"], "ir2": bad, "baseline_m": 0.05})
        except SystemExit:
            pass
        else:
            raise AssertionError("ppx distintos deben abortar")
    c = _cal()
    c["ir1"]["coeffs"] = [0.1, 0, 0, 0, 0]
    try:
        check_stereo_cal(c)
    except SystemExit:
        pass
    else:
        raise AssertionError("coeffs≠0 deben abortar")
    try:
        check_stereo_cal({"ir1": _cal()["ir1"], "ir2": _cal()["ir2"], "baseline_m": 0.0})
    except SystemExit:
        pass
    else:
        raise AssertionError("baseline 0 debe abortar")


def test_escala_efectiva_y_compat():
    with tempfile.TemporaryDirectory() as td:  # calibración vieja sin "model" sigue leyendo
        cal = _cal()
        (Path(td) / "cal.json").write_text(json.dumps(cal))
        disp = np.full((240, 320), 20.0, dtype=np.float32)  # mitad de resolución
        np.save(Path(td) / "d.npy", disp)
        z, valid, K, B = load_capture_depth(Path(td) / "d.npy", Path(td) / "cal.json", scale=1.0)
        assert K[0, 0] == 300.0 and z.shape == (240, 320), (K, z.shape)  # focal real, no la pedida
        assert valid.all() and abs(B - 0.055) < 1e-9


if __name__ == "__main__":
    test_rechaza_no_finitos_y_rango()
    test_calibracion_incompatible_aborta()
    test_escala_efectiva_y_compat()
    print("OK tests/test_depth.py: 3 checks")
