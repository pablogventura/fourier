"""Cuántos coeficientes se conservan, y en qué orden se dibujan.

El control pide una cantidad de componentes, de 1 a 500.

- Simétrica: se queda el mayor impar `2K+1` que no supera ese pedido, es decir
  las frecuencias de `-K` a `K`, DC incluido. Una cantidad par no se puede
  formar con pares ±k más el continuo, y se ajusta hacia abajo.
- Magnitud: los K coeficientes de mayor módulo, sin exigir simetría.

El orden de la suma no cambia el resultado. El orden del dibujo sí: primero
la frecuencia 0, después |k| creciente y, a igual módulo, la positiva antes
que la negativa.
"""

from __future__ import annotations

import numpy as np

from app.fourier.dft import HarmonicSeries


def snapped_component_count(strategy: str, component_count: int, available: int) -> int:
    count = int(max(1, min(component_count, available, 500)))
    if strategy == "symmetric" and count % 2 == 0:
        count -= 1
    return max(1, count)


def visual_key(frequency: int) -> tuple[int, int]:
    return (abs(int(frequency)), 0 if int(frequency) >= 0 else 1)


def select_series(series: HarmonicSeries, strategy: str, component_count: int) -> HarmonicSeries:
    if strategy not in {"symmetric", "magnitude"}:
        raise ValueError("La estrategia tiene que ser 'symmetric' o 'magnitude'.")
    count = snapped_component_count(strategy, component_count, len(series.coefficients))
    if strategy == "magnitude":
        ranking = sorted(
            range(len(series.coefficients)),
            key=lambda index: (
                -float(np.abs(series.coefficients[index])),
                abs(int(series.frequencies[index])),
                0 if int(series.frequencies[index]) >= 0 else 1,
            ),
        )
        chosen = ranking[:count]
    else:
        k_max = (count - 1) // 2
        chosen = [
            index
            for index in range(len(series.coefficients))
            if abs(int(series.frequencies[index])) <= k_max
        ]
    chosen.sort(key=lambda index: visual_key(int(series.frequencies[index])))
    return HarmonicSeries(
        frequencies=np.asarray([series.frequencies[index] for index in chosen], dtype=np.int64),
        coefficients=np.asarray([series.coefficients[index] for index in chosen], dtype=np.complex128),
        offset=series.offset,
    )
