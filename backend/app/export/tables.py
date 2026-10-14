"""CSV y JSON de coeficientes y de la curva de tasa-distorsión."""

from __future__ import annotations

import csv
import io
import json

import numpy as np


def coefficients_csv(records: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["frequency", "real", "imag", "amplitude", "phase"],
    )
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "frequency": record["frequency"],
                "real": f"{record['real']:.12g}",
                "imag": f"{record['imag']:.12g}",
                "amplitude": f"{record['amplitude']:.12g}",
                "phase": f"{record['phase']:.12g}",
            }
        )
    return buffer.getvalue()


def coefficients_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def metrics_csv(row: dict, curve: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["section", "key", "value"])
    for key, value in row.items():
        if key in {"payload", "reconstructed", "spectrum", "selected"}:
            continue
        writer.writerow(["summary", key, value])
    writer.writerow([])
    writer.writerow(
        ["component_count", "stored_coefficients", "encoded_bytes", "rmse", "max_error", "hausdorff"]
    )
    for item in curve:
        writer.writerow(
            [
                item["component_count"],
                item["stored_coefficients"],
                item["encoded_bytes"],
                f"{item['rmse']:.12g}",
                f"{item['max_error']:.12g}",
                f"{item['hausdorff']:.12g}",
            ]
        )
    return buffer.getvalue()


def records_from_series(frequencies: np.ndarray, coefficients: np.ndarray) -> list[dict]:
    records = []
    for frequency, coefficient in zip(frequencies, coefficients, strict=True):
        value = complex(coefficient)
        records.append(
            {
                "frequency": int(frequency),
                "real": float(np.real(value)),
                "imag": float(np.imag(value)),
                "amplitude": float(np.abs(value)),
                "phase": float(np.angle(value)),
            }
        )
    return records
