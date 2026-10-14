"""Figuras de ejemplo, en coordenadas de imagen (y hacia abajo).

El círculo, el cuadrado y la estrella recorren la pantalla en sentido
antihorario. El corazón usa las ecuaciones clásicas, con el eje vertical
invertido para que quede de pie. El colibrí es una silueta propia, suave,
para que pocos armónicos ya la hagan reconocible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.contours.resample import resample_contour
from app.contours.svg_paths import extract_svg_contours

CENTER = 200.0
RADIUS = 120.0


def _resample(vertices: np.ndarray, sample_count: int, closed: bool = True) -> np.ndarray:
    return resample_contour(vertices, sample_count, closed=closed)


def circle(sample_count: int = 512) -> np.ndarray:
    theta = 2.0 * np.pi * np.arange(sample_count) / sample_count
    x = CENTER + RADIUS * np.cos(theta)
    y = CENTER - RADIUS * np.sin(theta)
    return np.column_stack([x, y])


def square(sample_count: int = 512) -> np.ndarray:
    half = RADIUS
    c = CENTER
    vertices = np.array(
        [
            [c - half, c - half],
            [c - half, c + half],
            [c + half, c + half],
            [c + half, c - half],
        ],
        dtype=np.float64,
    )
    return _resample(vertices, sample_count, closed=True)


def star(sample_count: int = 512) -> np.ndarray:
    vertices = []
    for index in range(5):
        theta = np.pi / 2.0 + index * (4.0 * np.pi / 5.0)
        vertices.append(
            [CENTER + RADIUS * np.cos(theta), CENTER - RADIUS * np.sin(theta)]
        )
    return _resample(np.asarray(vertices, dtype=np.float64), sample_count, closed=True)


def heart(sample_count: int = 512) -> np.ndarray:
    t = 2.0 * np.pi * np.arange(4000) / 4000
    x = 16.0 * np.sin(t) ** 3
    y_up = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    raw = np.column_stack([x, -y_up])
    raw -= raw.mean(axis=0)
    span = max(float(np.max(np.abs(raw))), 1.0)
    raw *= RADIUS / span
    raw += CENTER
    return _resample(raw, sample_count, closed=True)


def hummingbird_svg() -> str:
    return Path(__file__).with_name("hummingbird.svg").read_text(encoding="utf-8")


def hummingbird(sample_count: int = 512) -> np.ndarray:
    contours, _warnings = extract_svg_contours(hummingbird_svg(), sample_count)
    if not contours:
        raise RuntimeError("La silueta del colibrí no produjo un contorno.")
    closed = [item for item in contours if item["closed"]] or contours
    best = max(closed, key=lambda item: item["perimeter"])
    return np.asarray(best["points"], dtype=np.float64)


PRESETS = {
    "circle": {
        "title": "Círculo",
        "note": "Un círculo antihorario tiene un solo coeficiente, el de frecuencia +1.",
        "builder": circle,
        "closed": True,
    },
    "square": {
        "title": "Cuadrado",
        "note": "Las esquinas decaen lento. Hacen falta muchos armónicos y el corte se ve en los vértices.",
        "builder": square,
        "closed": True,
    },
    "star": {
        "title": "Estrella",
        "note": "La polilínea se cruza. El parámetro sigue el trazo, no el borde exterior de la figura.",
        "builder": star,
        "closed": True,
    },
    "heart": {
        "title": "Corazón",
        "note": "Curva clásica, suave. Con pocas decenas de coeficientes ya se reconoce.",
        "builder": heart,
        "closed": True,
    },
    "hummingbird": {
        "title": "Colibrí",
        "note": "Silueta cerrada de pocos tramos. Comparala con el cuadrado: acá el espectro cae más rápido.",
        "builder": hummingbird,
        "closed": True,
    },
}


def list_presets() -> list[dict]:
    return [{"id": key, "title": value["title"], "note": value["note"]} for key, value in PRESETS.items()]


def build_preset(name: str, sample_count: int) -> tuple[np.ndarray, bool, str]:
    if name not in PRESETS:
        raise ValueError(f"No existe el ejemplo '{name}'.")
    preset = PRESETS[name]
    points = preset["builder"](sample_count)
    return points, bool(preset["closed"]), str(preset["note"])
