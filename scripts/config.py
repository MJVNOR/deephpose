"""Defaults centrales desde config.toml (raíz). El CLI siempre gana."""
import sys
import tomllib
from pathlib import Path


def load():
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / "config.toml").is_file():
            with open(base / "config.toml", "rb") as f:
                return tomllib.load(f)
    sys.exit("error: config.toml no encontrado (corre desde la raíz del proyecto)")
