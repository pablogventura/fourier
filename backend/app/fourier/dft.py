"""DFT de un contorno con la normalización de NumPy y frecuencias firmadas.

Las muestras viven en coordenadas de imagen: x hacia la derecha, y hacia abajo.
Para que una frecuencia positiva gire en sentido antihorario en la pantalla,
la señal compleja es

    z[n] = x[n] - i y[n]

Así la parte imaginaria apunta hacia arriba. En instantes enteros t = n/N,
el bin N-1 y la frecuencia -1 coinciden. Entre muestras, no. La animación
usa siempre la frecuencia firmada. Si N es par, el bin de Nyquist se etiqueta
como +N/2 y no se duplica: en las muestras enteras da lo mismo que -N/2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarmonicSeries:
    frequencies: np.ndarray
    coefficients: np.ndarray
    offset: float


@dataclass(frozen=True)
class ContourSpectrum:
    sample_count: int
    centered: bool
    offset_x: float
    offset_y: float
    complex: HarmonicSeries
    x: HarmonicSeries
    y: HarmonicSeries


def signed_frequencies(sample_count: int) -> np.ndarray:
    """Frecuencias en ciclos por período, en el orden de `numpy.fft.fft`."""
    bins = np.arange(sample_count, dtype=np.int64)
    signed = np.where(bins < sample_count / 2, bins, bins - sample_count)
    if sample_count % 2 == 0:
        signed[sample_count // 2] = sample_count // 2
    return signed


def _series_from_samples(samples: np.ndarray, center: bool) -> HarmonicSeries:
    values = np.asarray(samples)
    offset = complex(np.mean(values)) if center else 0j
    centered = values - offset
    coefficients = np.fft.fft(centered) / len(centered)
    return HarmonicSeries(
        frequencies=signed_frequencies(len(centered)),
        coefficients=coefficients.astype(np.complex128),
        offset=offset,
    )


def analyze_complex(samples: np.ndarray, center: bool = False) -> ContourSpectrum:
    """Analiza una señal ya expresada como z = x - i y, o cualquier z de prueba."""
    z = np.asarray(samples, dtype=np.complex128)
    if z.ndim != 1 or len(z) < 4:
        raise ValueError("La señal compleja necesita al menos 4 muestras.")
    series = _series_from_samples(z, center=center)
    # offset de analyze_complex es el promedio de z, no un (x, y) de imagen.
    offset_x = float(np.real(series.offset))
    offset_y = float(-np.imag(series.offset))
    x_series = _series_from_samples(np.real(z), center=center)
    y_series = _series_from_samples(-np.imag(z), center=center)
    return ContourSpectrum(
        sample_count=len(z),
        centered=center,
        offset_x=offset_x,
        offset_y=offset_y,
        complex=series,
        x=x_series,
        y=y_series,
    )


def analyze_source_points(points: np.ndarray, center: bool = True) -> ContourSpectrum:
    """FFT de una polilínea en coordenadas de imagen (y hacia abajo)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        raise ValueError("Hacen falta al menos 4 puntos (x, y).")
    z = pts[:, 0] - 1j * pts[:, 1]
    spectrum = analyze_complex(z, center=center)
    if center:
        spectrum = ContourSpectrum(
            sample_count=spectrum.sample_count,
            centered=True,
            offset_x=float(np.mean(pts[:, 0])),
            offset_y=float(np.mean(pts[:, 1])),
            complex=spectrum.complex,
            x=spectrum.x,
            y=spectrum.y,
        )
    else:
        spectrum = ContourSpectrum(
            sample_count=spectrum.sample_count,
            centered=False,
            offset_x=0.0,
            offset_y=0.0,
            complex=spectrum.complex,
            x=spectrum.x,
            y=spectrum.y,
        )
    return spectrum


def coefficient_records(series: HarmonicSeries) -> list[dict]:
    records = []
    order = np.argsort(series.frequencies, kind="mergesort")
    for index in order:
        coefficient = complex(series.coefficients[index])
        frequency = int(series.frequencies[index])
        records.append(
            {
                "frequency": frequency,
                "real": float(np.real(coefficient)),
                "imag": float(np.imag(coefficient)),
                "amplitude": float(np.abs(coefficient)),
                "phase": float(np.angle(coefficient)),
            }
        )
    return records
