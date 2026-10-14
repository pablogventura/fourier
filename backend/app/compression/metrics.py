"""Tamaños y errores de un contorno guardado como serie de Fourier.

Los bytes crudos cuentan muestras o coeficientes en float64, sin cabecera.
El archivo FLC1 es un contenedor: tiene marca, escalas y el tipo numérico.
Esas dos cuentas no se mezclan. Además, esto comprime una curva, no el mapa
de bits del que salió.
"""

from __future__ import annotations

import numpy as np

from app.compression.flc1 import encode_flc1
from app.fourier.dft import ContourSpectrum, analyze_source_points
from app.fourier.reconstruct import reconstruct_source
from app.fourier.select import select_series, snapped_component_count


def directed_hausdorff(source: np.ndarray, target: np.ndarray) -> float:
    worst = 0.0
    step = 256
    for start in range(0, len(source), step):
        chunk = source[start : start + step]
        diff = chunk[:, None, :] - target[None, :, :]
        distance = np.linalg.norm(diff, axis=2)
        worst = max(worst, float(distance.min(axis=1).max()))
    return worst


def hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    return max(directed_hausdorff(first, second), directed_hausdorff(second, first))


def reconstruction_errors(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    delta = np.linalg.norm(original - reconstructed, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "max_error": float(np.max(delta)),
        "hausdorff": hausdorff(original, reconstructed),
    }


def evaluate_compression(
    points: np.ndarray,
    closed: bool,
    center: bool,
    strategy: str,
    component_count: int,
    number_format: str,
    quant_step: float,
) -> dict:
    spectrum = analyze_source_points(points, center=center)
    selected = select_series(spectrum.complex, strategy, component_count)
    payload = encode_flc1(
        selected,
        spectrum.sample_count,
        spectrum.offset_x,
        spectrum.offset_y,
        closed=closed,
        centered=center,
        number_format=number_format,
        quant_step=quant_step,
    )
    from app.compression.flc1 import decode_flc1

    document = decode_flc1(payload)
    reconstructed = document.reconstruct()
    errors = reconstruction_errors(np.asarray(points, dtype=np.float64), reconstructed)
    sample_count = spectrum.sample_count
    raw_points_bytes = sample_count * 2 * 8
    raw_coefficients_bytes = sample_count * 2 * 8
    encoded_bytes = len(payload)
    kept = snapped_component_count(strategy, component_count, sample_count)
    return {
        "sample_count": sample_count,
        "component_count": kept,
        "stored_coefficients": int(len(selected.frequencies)),
        "number_format": number_format,
        "quant_step": float(quant_step) if number_format == "int16" else 0.0,
        "raw_points_bytes": raw_points_bytes,
        "raw_coefficients_bytes": raw_coefficients_bytes,
        "encoded_bytes": encoded_bytes,
        "ratio_vs_points": raw_points_bytes / encoded_bytes if encoded_bytes else None,
        "ratio_vs_coefficients": raw_coefficients_bytes / encoded_bytes if encoded_bytes else None,
        "rmse": errors["rmse"],
        "max_error": errors["max_error"],
        "hausdorff": errors["hausdorff"],
        "payload": payload,
        "reconstructed": reconstructed,
        "spectrum": spectrum,
        "selected": selected,
    }


def rate_distortion(
    points: np.ndarray,
    closed: bool,
    center: bool,
    strategy: str,
    number_format: str,
    quant_step: float,
    focus_count: int | None = None,
) -> list[dict]:
    sample_count = len(points)
    cap = min(sample_count, 500)
    if cap <= 1:
        grid = [1]
    else:
        raw = np.geomspace(1, cap, num=min(36, cap))
        grid = {1, cap, *(int(round(value)) for value in raw)}
        if focus_count is not None:
            grid.add(int(focus_count))
    counts = sorted(
        {
            snapped_component_count(strategy, value, sample_count)
            for value in grid
            if value >= 1
        }
    )
    rows = []
    for count in counts:
        report = evaluate_compression(
            points,
            closed=closed,
            center=center,
            strategy=strategy,
            component_count=count,
            number_format=number_format,
            quant_step=quant_step,
        )
        rows.append(
            {
                "component_count": report["component_count"],
                "stored_coefficients": report["stored_coefficients"],
                "encoded_bytes": report["encoded_bytes"],
                "rmse": report["rmse"],
                "max_error": report["max_error"],
                "hausdorff": report["hausdorff"],
            }
        )
    return rows


def full_reference_spectrum(points: np.ndarray, center: bool) -> ContourSpectrum:
    return analyze_source_points(points, center=center)


def reference_curve(series, sample_count: int, offset_x: float, offset_y: float) -> np.ndarray:
    return reconstruct_source(series, sample_count, offset_x, offset_y)
