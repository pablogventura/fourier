"""Remuestreo uniforme por longitud de arco.

Los puntos de entrada pueden estar agrupados. La serie de Fourier usa el
índice de muestra como parámetro, así que una densidad irregular se leería
como una velocidad variable. Acá el parámetro queda proporcional al arco.
"""

from __future__ import annotations

import numpy as np


def dedupe_points(points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """Quita vértices consecutivos repetidos."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts.reshape(0, 2)
    keep = [0]
    for index in range(1, len(pts)):
        if np.linalg.norm(pts[index] - pts[keep[-1]]) > tolerance:
            keep.append(index)
    return pts[np.array(keep, dtype=int)]


def prepare_polyline(points: np.ndarray, closed: bool) -> np.ndarray:
    """Normaliza una polilínea y, si está cerrada, no repite el vértice final."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Se esperaban puntos con forma (N, 2).")
    if len(pts) < 2:
        raise ValueError("Un contorno necesita al menos dos puntos.")
    if closed and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    pts = dedupe_points(pts)
    minimum = 3 if closed else 2
    if len(pts) < minimum:
        raise ValueError("El contorno queda degenerado después de quitar puntos repetidos.")
    return pts


def arc_length_table(points: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve vértices y la longitud acumulada, incluido el cierre si corresponde."""
    pts = prepare_polyline(points, closed)
    if closed:
        nxt = np.roll(pts, -1, axis=0)
        segment = np.linalg.norm(nxt - pts, axis=1)
    else:
        segment = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    if cumulative[-1] <= 1e-12:
        raise ValueError("La longitud del contorno es nula.")
    return pts, cumulative


def resample_contour(points: np.ndarray, sample_count: int, closed: bool) -> np.ndarray:
    """Distribuye `sample_count` puntos con arcos iguales.

    En un contorno cerrado el último punto no repite al primero: el período
    está implícito. En uno abierto se conservan los extremos.
    """
    if sample_count < 4:
        raise ValueError("Hacen falta al menos 4 muestras.")
    pts, cumulative = arc_length_table(points, closed)
    total = float(cumulative[-1])
    if closed:
        targets = np.linspace(0.0, total, sample_count, endpoint=False)
        vertex_s = cumulative[:-1]
        segment_index = np.searchsorted(vertex_s, targets, side="right") - 1
        segment_index = np.clip(segment_index, 0, len(pts) - 1)
        start = pts[segment_index]
        end = pts[(segment_index + 1) % len(pts)]
        segment_length = cumulative[segment_index + 1] - cumulative[segment_index]
    else:
        targets = np.linspace(0.0, total, sample_count, endpoint=True)
        segment_index = np.searchsorted(cumulative, targets, side="right") - 1
        segment_index = np.clip(segment_index, 0, len(pts) - 2)
        start = pts[segment_index]
        end = pts[segment_index + 1]
        segment_length = cumulative[segment_index + 1] - cumulative[segment_index]
    local = np.divide(
        targets - cumulative[segment_index],
        segment_length,
        out=np.zeros(sample_count, dtype=np.float64),
        where=segment_length > 1e-15,
    )
    sampled = start + (end - start) * local[:, None]
    if not closed:
        sampled[0] = pts[0]
        sampled[-1] = pts[-1]
    return sampled


def perimeter(points: np.ndarray, closed: bool) -> float:
    _pts, cumulative = arc_length_table(points, closed)
    return float(cumulative[-1])
