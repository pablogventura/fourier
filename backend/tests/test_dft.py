"""Convención de la FFT: signo, normalización y tiempo continuo."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from app.fourier.dft import analyze_complex, analyze_source_points, signed_frequencies
from app.fourier.phasors import math_chain
from app.fourier.reconstruct import reconstruct_complex
from app.fourier.select import select_series
from app.presets.shapes import circle


def _coeff(spectrum, frequency: int) -> complex:
    match = np.where(spectrum.complex.frequencies == frequency)[0]
    assert len(match) == 1
    return complex(spectrum.complex.coefficients[match[0]])


def test_signed_nyquist_is_positive_and_unique():
    signed = signed_frequencies(8)
    assert list(signed) == [0, 1, 2, 3, 4, -3, -2, -1]
    assert list(signed).count(4) == 1
    assert -4 not in signed


def test_counterclockwise_circle_is_positive_frequency():
    n = 64
    theta = 2 * np.pi * np.arange(n) / n
    spectrum = analyze_complex(np.exp(1j * theta), center=False)
    assert abs(_coeff(spectrum, 1) - 1) < 1e-9
    for frequency in spectrum.complex.frequencies:
        if int(frequency) != 1:
            assert abs(_coeff(spectrum, int(frequency))) < 1e-9


def test_clockwise_circle_uses_negative_frequency_between_samples():
    n = 6
    theta = 2 * np.pi * np.arange(n) / n
    spectrum = analyze_complex(np.exp(-1j * theta), center=False)
    assert abs(_coeff(spectrum, -1) - 1) < 1e-9
    selected = select_series(spectrum.complex, "magnitude", 1)
    at = reconstruct_complex(selected, np.array([0.25]))[0]
    # Con el bin sin firmar (frecuencia N-1 = 5) este instante caería en +i.
    assert abs(at - (-1j)) < 1e-9
    angles = np.angle(reconstruct_complex(selected, np.array([0.0, 0.1])))
    assert angles[1] < angles[0]


def test_two_tone_signal_keeps_only_those_bins():
    n = 128
    step = np.arange(n)
    signal = np.exp(2j * np.pi * step / n) + 0.3 * np.exp(-2j * np.pi * 3 * step / n)
    spectrum = analyze_complex(signal, center=False)
    assert abs(_coeff(spectrum, 1) - 1) < 1e-9
    assert abs(_coeff(spectrum, -3) - 0.3) < 1e-9
    others = [
        abs(_coeff(spectrum, int(frequency)))
        for frequency in spectrum.complex.frequencies
        if int(frequency) not in {1, -3}
    ]
    assert max(others) < 1e-9


def test_full_reconstruction_matches_samples_and_is_periodic():
    n = 96
    step = np.arange(n)
    signal = np.exp(2j * np.pi * step / n) + 0.2 * np.exp(-2j * np.pi * 4 * step / n) + 0.05
    spectrum = analyze_complex(signal, center=False)
    rebuilt = reconstruct_complex(spectrum.complex, step / n)
    assert np.max(np.abs(rebuilt - signal)) < 1e-8
    ends = reconstruct_complex(spectrum.complex, np.array([0.0, 1.0]))
    assert abs(ends[0] - ends[1]) < 1e-8


def test_source_circle_preset_is_frequency_one_after_centering():
    points = circle(128)
    spectrum = analyze_source_points(points, center=True)
    assert abs(spectrum.offset_x - 200) < 1e-8
    assert abs(spectrum.offset_y - 200) < 1e-8
    assert abs(_coeff(spectrum, 0)) < 1e-8
    assert abs(_coeff(spectrum, 1) - 120) < 1e-6


def test_centering_moves_translation_into_the_offset():
    n = 64
    theta = 2 * np.pi * np.arange(n) / n
    points = np.column_stack([50 + np.cos(theta), -20 - np.sin(theta)])
    centered = analyze_source_points(points, center=True)
    raw = analyze_source_points(points, center=False)
    assert abs(_coeff(centered, 0)) < 1e-8
    assert abs(centered.offset_x - 50) < 1e-8
    assert abs(centered.offset_y - (-20)) < 1e-8
    assert abs(_coeff(raw, 0)) > 10


def test_complex_coefficients_match_separate_axes():
    points = circle(64)
    spectrum = analyze_source_points(points, center=True)
    mixed = spectrum.x.coefficients - 1j * spectrum.y.coefficients
    assert np.max(np.abs(mixed - spectrum.complex.coefficients)) < 1e-8


def test_math_chain_order_for_the_two_tone():
    n = 32
    step = np.arange(n)
    signal = np.exp(2j * np.pi * step / n) + 0.3 * np.exp(-2j * np.pi * 3 * step / n)
    spectrum = analyze_complex(signal, center=False)
    selected = select_series(spectrum.complex, "magnitude", 2)
    tips = math_chain(selected, 0.25)
    assert abs(tips[0][0] - 0) < 1e-8
    assert abs(tips[0][1] - 1) < 1e-8
    assert abs(tips[1][0] - 0) < 1e-8
    assert abs(tips[1][1] - 1.3) < 1e-8


def _square_math(t: float) -> complex:
    """Cuadrado de lado 2 centrado, recorrido antihorario en el plano matemático."""
    local = t % 1.0
    if local < 0.25:
        return 1 + 1j * (-1 + 8 * local)
    if local < 0.5:
        return (3 - 8 * local) + 1j
    if local < 0.75:
        return -1 + 1j * (5 - 8 * local)
    return (8 * local - 7) - 1j


def test_square_matches_edge_integral():
    n = 4096
    samples = np.array([_square_math(index / n) for index in range(n)])
    spectrum = analyze_complex(samples, center=False)

    def integrate(frequency: int) -> complex:
        def real_part(t: float) -> float:
            return float(np.real(_square_math(t) * np.exp(-2j * np.pi * frequency * t)))

        def imag_part(t: float) -> float:
            return float(np.imag(_square_math(t) * np.exp(-2j * np.pi * frequency * t)))

        real, _real_err = quad(real_part, 0, 1, epsabs=1e-10, limit=200)
        imag, _imag_err = quad(imag_part, 0, 1, epsabs=1e-10, limit=200)
        return real + 1j * imag

    for frequency in range(-5, 6):
        assert abs(_coeff(spectrum, frequency) - integrate(frequency)) < 2e-3
