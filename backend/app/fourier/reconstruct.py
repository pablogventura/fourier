"""Reconstrucción en tiempo continuo a partir de coeficientes ya elegidos.

    z(t) = Σ c_k exp(2 π i k t)

t = 0 y t = 1 son el mismo punto, porque k es entero. No hace falta volver a
calcular la FFT para cambiar K: se filtran los coeficientes y se suma de nuevo.
"""

from __future__ import annotations

import numpy as np

from app.fourier.dft import HarmonicSeries


def reconstruct_complex(series: HarmonicSeries, t: np.ndarray) -> np.ndarray:
    times = np.asarray(t, dtype=np.float64).reshape(-1)
    if len(series.frequencies) == 0:
        return np.zeros(len(times), dtype=np.complex128)
    phases = np.exp(
        2j
        * np.pi
        * series.frequencies.astype(np.float64)[:, None]
        * times[None, :]
    )
    return series.coefficients.astype(np.complex128) @ phases


def math_to_source(
    values: np.ndarray,
    offset_x: float,
    offset_y: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """Pasa z = x_math + i y_math a pixeles, con y de imagen hacia abajo."""
    x = offset_x + scale_x * np.real(values)
    y = offset_y - scale_y * np.imag(values)
    return np.column_stack([x, y])


def reconstruct_source(
    series: HarmonicSeries,
    sample_count: int,
    offset_x: float,
    offset_y: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    t = np.arange(sample_count, dtype=np.float64) / sample_count
    values = reconstruct_complex(series, t)
    return math_to_source(values, offset_x, offset_y, scale_x, scale_y)


def reconstruct_real(series: HarmonicSeries, t: np.ndarray, offset: float) -> np.ndarray:
    """Parte real de una serie. Para x(t) o y(t) por separado."""
    return offset + np.real(reconstruct_complex(series, t))
