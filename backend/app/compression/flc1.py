"""Archivo binario FLC1: un contorno guardado solo con coeficientes de Fourier.

Little-endian. No guarda la imagen ni los puntos. La curva se reconstruye en
N instantes uniformes del período.

    source_x = offset_x + scale_x * Re(z)
    source_y = offset_y - scale_y * Im(z)
    z(t) = Σ c_k exp(2 π i k t)

Si el flag de centrado está activo, el offset es el centroide en coordenadas
de imagen y los coeficientes describen la forma. Si no, el offset es cero y
la traslación vive en el término de continua.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from app.fourier.dft import HarmonicSeries
from app.fourier.reconstruct import reconstruct_source

MAGIC = b"FLC1"
VERSION = 1
HEADER = struct.Struct("<4sHHIIddddd")
RECORD_F64 = struct.Struct("<idd")
RECORD_F32 = struct.Struct("<iff")
RECORD_I16 = struct.Struct("<ihh")

FLAG_CLOSED = 1
FLAG_QUANTIZED = 2
FLAG_FLOAT64 = 4
FLAG_CENTERED = 8

HEADER_BYTES = HEADER.size


@dataclass(frozen=True)
class Flc1Document:
    sample_count: int
    closed: bool
    centered: bool
    number_format: str
    quant_step: float
    offset_x: float
    offset_y: float
    scale_x: float
    scale_y: float
    frequencies: np.ndarray
    coefficients: np.ndarray

    @property
    def series(self) -> HarmonicSeries:
        return HarmonicSeries(self.frequencies, self.coefficients, offset=0j)

    def reconstruct(self) -> np.ndarray:
        return reconstruct_source(
            self.series,
            self.sample_count,
            self.offset_x,
            self.offset_y,
            self.scale_x,
            self.scale_y,
        )


def quantize_coefficients(coefficients: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if step <= 0:
        raise ValueError("El paso de cuantización tiene que ser positivo.")
    real_q = np.clip(np.round(np.real(coefficients) / step), -32768, 32767).astype(np.int16)
    imag_q = np.clip(np.round(np.imag(coefficients) / step), -32768, 32767).astype(np.int16)
    restored = real_q.astype(np.float64) * step + 1j * imag_q.astype(np.float64) * step
    return restored, real_q, imag_q


def encode_flc1(
    series: HarmonicSeries,
    sample_count: int,
    offset_x: float,
    offset_y: float,
    closed: bool,
    centered: bool,
    number_format: str = "float32",
    quant_step: float = 0.05,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> bytes:
    if number_format not in {"float64", "float32", "int16"}:
        raise ValueError("El formato numérico tiene que ser float64, float32 o int16.")
    frequencies = np.asarray(series.frequencies, dtype=np.int64)
    coefficients = np.asarray(series.coefficients, dtype=np.complex128)
    flags = 0
    if closed:
        flags |= FLAG_CLOSED
    if centered:
        flags |= FLAG_CENTERED
    stored_step = 0.0
    real_q = imag_q = None
    if number_format == "int16":
        flags |= FLAG_QUANTIZED
        stored_step = float(quant_step)
        _restored, real_q, imag_q = quantize_coefficients(coefficients, stored_step)
    elif number_format == "float64":
        flags |= FLAG_FLOAT64

    header = HEADER.pack(
        MAGIC,
        VERSION,
        flags,
        int(sample_count),
        int(len(frequencies)),
        float(offset_x),
        float(offset_y),
        float(scale_x),
        float(scale_y),
        float(stored_step),
    )
    chunks = [header]
    for index, frequency in enumerate(frequencies):
        if number_format == "int16":
            chunks.append(RECORD_I16.pack(int(frequency), int(real_q[index]), int(imag_q[index])))
        elif number_format == "float64":
            chunks.append(
                RECORD_F64.pack(
                    int(frequency),
                    float(np.real(coefficients[index])),
                    float(np.imag(coefficients[index])),
                )
            )
        else:
            chunks.append(
                RECORD_F32.pack(
                    int(frequency),
                    float(np.float32(np.real(coefficients[index]))),
                    float(np.float32(np.imag(coefficients[index]))),
                )
            )
    return b"".join(chunks)


def decode_flc1(payload: bytes) -> Flc1Document:
    if len(payload) < HEADER_BYTES:
        raise ValueError("El archivo es más corto que la cabecera FLC1.")
    magic, version, flags, sample_count, count, offset_x, offset_y, scale_x, scale_y, quant_step = HEADER.unpack_from(
        payload, 0
    )
    if magic != MAGIC:
        raise ValueError("El archivo no empieza con la marca FLC1.")
    if version != VERSION:
        raise ValueError(f"Versión de FLC1 no reconocida: {version}.")
    quantized = bool(flags & FLAG_QUANTIZED)
    use_float64 = bool(flags & FLAG_FLOAT64)
    if quantized:
        record = RECORD_I16
        number_format = "int16"
    elif use_float64:
        record = RECORD_F64
        number_format = "float64"
    else:
        record = RECORD_F32
        number_format = "float32"
    expected = HEADER_BYTES + count * record.size
    if len(payload) < expected:
        raise ValueError("El archivo FLC1 está truncado.")
    frequencies = np.empty(count, dtype=np.int64)
    coefficients = np.empty(count, dtype=np.complex128)
    cursor = HEADER_BYTES
    for index in range(count):
        unpacked = record.unpack_from(payload, cursor)
        cursor += record.size
        frequencies[index] = int(unpacked[0])
        if quantized:
            coefficients[index] = (int(unpacked[1]) + 1j * int(unpacked[2])) * float(quant_step)
        else:
            coefficients[index] = complex(float(unpacked[1]), float(unpacked[2]))
    return Flc1Document(
        sample_count=int(sample_count),
        closed=bool(flags & FLAG_CLOSED),
        centered=bool(flags & FLAG_CENTERED),
        number_format=number_format,
        quant_step=float(quant_step),
        offset_x=float(offset_x),
        offset_y=float(offset_y),
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        frequencies=frequencies,
        coefficients=coefficients,
    )
