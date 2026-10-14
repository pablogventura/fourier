"""Remuestreo, truncamiento simétrico y selección por magnitud."""

from __future__ import annotations

import numpy as np

from app.contours.resample import resample_contour
from app.fourier.dft import analyze_complex
from app.fourier.reconstruct import reconstruct_complex
from app.fourier.select import select_series, snapped_component_count


def test_resampling_equalizes_chord_lengths_on_a_circle():
    line = np.array([[0.0, 0.0], [0.1, 0.0], [0.4, 0.0], [6.0, 0.0], [10.0, 0.0]])
    straight = resample_contour(line, 21, closed=False)
    steps = np.diff(straight[:, 0])
    assert np.max(np.abs(steps - steps[0])) < 1e-9

    warped = np.linspace(0, 1, 4000, endpoint=False) ** 1.35
    warped *= 2 * np.pi / warped[-1]
    raw = np.column_stack([np.cos(warped), np.sin(warped)])
    sampled = resample_contour(raw, 256, closed=True)
    chords = np.linalg.norm(np.roll(sampled, -1, axis=0) - sampled, axis=1)
    assert np.std(chords) / np.mean(chords) < 2e-3
    delta = np.diff(np.unwrap(np.arctan2(sampled[:, 1], sampled[:, 0])))
    assert np.std(delta) / np.mean(np.abs(delta)) < 2e-2


def test_open_resample_keeps_endpoints():
    raw = np.array([[0.0, 0.0], [0.2, 1.0], [3.0, 1.2], [4.0, 0.0]])
    sampled = resample_contour(raw, 16, closed=False)
    assert np.allclose(sampled[0], raw[0])
    assert np.allclose(sampled[-1], raw[-1])


def test_symmetric_count_snaps_to_odd_and_magnitude_differs():
    assert snapped_component_count("symmetric", 4, 64) == 3
    n = 64
    step = np.arange(n)
    signal = np.exp(2j * np.pi * step / n) + 0.5 * np.exp(2j * np.pi * 10 * step / n)
    spectrum = analyze_complex(signal, center=False)
    symmetric = select_series(spectrum.complex, "symmetric", 5)
    magnitude = select_series(spectrum.complex, "magnitude", 2)
    assert set(int(k) for k in symmetric.frequencies) == {-2, -1, 0, 1, 2}
    assert set(int(k) for k in magnitude.frequencies) == {1, 10}
    samples = step / n
    original = signal
    err_sym = np.max(np.abs(reconstruct_complex(symmetric, samples) - original))
    err_mag = np.max(np.abs(reconstruct_complex(magnitude, samples) - original))
    assert err_sym > err_mag


def test_visual_order_puts_dc_then_positive_before_negative():
    n = 16
    spectrum = analyze_complex(np.exp(2j * np.pi * np.arange(n) / n), center=False)
    selected = select_series(spectrum.complex, "symmetric", 5)
    assert list(int(k) for k in selected.frequencies) == [0, 1, -1, 2, -2]
