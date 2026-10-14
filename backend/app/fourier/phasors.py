"""Posiciones de la cadena de epiciclos en coordenadas de imagen.

La suma compleja vive con el imaginario hacia arriba. Al dibujar, la
componente vertical cambia de signo para que +y siga hacia abajo, igual que
la imagen, y una frecuencia positiva recorra la pantalla en antihorario.
"""

from __future__ import annotations

import numpy as np

from app.fourier.dft import HarmonicSeries


def source_chain(
    series: HarmonicSeries,
    t: float,
    offset_x: float,
    offset_y: float,
) -> list[dict]:
    """Círculos encadenados. Cada ítem es el círculo cuyo extremo es el nuevo origen."""
    frequencies = series.frequencies.astype(np.float64)
    coefficients = series.coefficients.astype(np.complex128)
    theta = np.angle(coefficients) + 2.0 * np.pi * frequencies * float(t)
    amplitude = np.abs(coefficients)
    dx = amplitude * np.cos(theta)
    dy = -amplitude * np.sin(theta)
    cursor_x = float(offset_x)
    cursor_y = float(offset_y)
    circles = []
    for index in range(len(amplitude)):
        nxt_x = cursor_x + float(dx[index])
        nxt_y = cursor_y + float(dy[index])
        circles.append(
            {
                "cx": cursor_x,
                "cy": cursor_y,
                "x": nxt_x,
                "y": nxt_y,
                "radius": float(amplitude[index]),
                "frequency": int(series.frequencies[index]),
            }
        )
        cursor_x, cursor_y = nxt_x, nxt_y
    return circles


def math_chain(series: HarmonicSeries, t: float) -> list[tuple[float, float]]:
    """Extremos acumulados en el plano complejo, sin el cambio de signo de y."""
    cursor = 0j
    tips: list[tuple[float, float]] = []
    for frequency, coefficient in zip(series.frequencies, series.coefficients, strict=True):
        theta = float(np.angle(coefficient) + 2.0 * np.pi * float(frequency) * float(t))
        amplitude = float(np.abs(coefficient))
        cursor += amplitude * np.exp(1j * theta)
        tips.append((float(np.real(cursor)), float(np.imag(cursor))))
    return tips
