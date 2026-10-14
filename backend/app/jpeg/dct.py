"""DCT 8x8 en la escala ortonormal, que para N=8 coincide con la del JPEG.

Eso permite usar la tabla de luminancia del estándar y la escala de calidad
del Independent JPEG Group, y ver un efecto parecido al de un JPEG. No es un
codificador JPEG: no hay YCbCr, no hay submuestreo 4:2:0 y no hay Huffman ni
aritmética.

Antes de la DCT de una imagen se resta 128, como en el estándar, para centrar
el rango 0..255. El test del bloque constante usa la DCT sin ese desplazamiento:
un bloque lleno de `c` tiene coeficiente DC igual a `8c`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import convolve

LUMINANCE_Q50 = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)


def dct2(block: np.ndarray) -> np.ndarray:
    return dctn(np.asarray(block, dtype=np.float64), type=2, norm="ortho")


def idct2(coefficients: np.ndarray) -> np.ndarray:
    return idctn(np.asarray(coefficients, dtype=np.float64), type=2, norm="ortho")


def quality_table(quality: int) -> np.ndarray:
    level = int(np.clip(quality, 1, 100))
    if level < 50:
        scale = 5000.0 / level
    else:
        scale = 200.0 - 2.0 * level
    table = np.floor((LUMINANCE_Q50 * scale + 50.0) / 100.0)
    return np.clip(table, 1.0, 255.0)


def basis_patterns() -> list[list[list[float]]]:
    """64 patrones, fila v y columna u, en el orden v * 8 + u."""
    samples = np.arange(8, dtype=np.float64)
    vectors = []
    for frequency in range(8):
        if frequency == 0:
            vectors.append(np.ones(8) / np.sqrt(8.0))
        else:
            vectors.append(
                np.sqrt(2.0 / 8.0) * np.cos(np.pi * frequency * (2 * samples + 1) / 16.0)
            )
    patterns = []
    for v in range(8):
        for u in range(8):
            patterns.append(np.outer(vectors[v], vectors[u]).tolist())
    return patterns


def pad_to_blocks(rgb: np.ndarray, max_side: int = 384) -> np.ndarray:
    from PIL import Image

    from app.contours.raster import _limit_side

    image = _limit_side(Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB"), max_side)
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return array


@dataclass(frozen=True)
class BlockEdit:
    u: int
    v: int
    value: float


def _suppress_high(coefficients: np.ndarray, cutoff: int) -> None:
    if cutoff >= 14:
        return
    for v in range(8):
        for u in range(8):
            if u + v > cutoff:
                coefficients[v, u] = 0.0


def process_plane(
    plane: np.ndarray,
    table: np.ndarray,
    cutoff: int,
    edits: list[BlockEdit],
    block_row: int,
    block_col: int,
    apply_edits: bool,
) -> tuple[np.ndarray, dict | None]:
    height, width = plane.shape
    restored = np.zeros((height, width), dtype=np.float64)
    selected: dict | None = None
    rows = height // 8
    cols = width // 8
    block_row = int(np.clip(block_row, 0, max(0, rows - 1)))
    block_col = int(np.clip(block_col, 0, max(0, cols - 1)))
    for row in range(rows):
        for col in range(cols):
            window = plane[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8].astype(np.float64)
            original = dct2(window - 128.0)
            quantized = np.round(original / table)
            dequant = quantized * table
            _suppress_high(dequant, cutoff)
            if apply_edits and row == block_row and col == block_col:
                for edit in edits:
                    if 0 <= edit.u < 8 and 0 <= edit.v < 8:
                        dequant[edit.v, edit.u] = float(edit.value)
            if row == block_row and col == block_col:
                selected = {
                    "row": block_row,
                    "col": block_col,
                    "original": original.tolist(),
                    "quantized": quantized.tolist(),
                    "dequantized": dequant.tolist(),
                }
            spatial = idct2(dequant) + 128.0
            restored[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] = spatial
    clipped = np.clip(np.round(restored), 0, 255).astype(np.uint8)
    return clipped, selected


def reconstruct_jpeg(
    rgb: np.ndarray,
    quality: int,
    cutoff: int,
    grayscale: bool,
    block_row: int,
    block_col: int,
    channel: int,
    edits: list[BlockEdit],
) -> dict:
    table = quality_table(quality)
    channel = int(np.clip(channel, 0, 2))
    if grayscale:
        gray = np.round(
            0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        ).astype(np.uint8)
        restored, selected = process_plane(
            gray, table, cutoff, edits, block_row, block_col, apply_edits=True
        )
        original = np.repeat(gray[:, :, None], 3, axis=2)
        reconstructed = np.repeat(restored[:, :, None], 3, axis=2)
        metric_a, metric_b = gray, restored
    else:
        planes = []
        selected = None
        for index in range(3):
            plane, block = process_plane(
                rgb[:, :, index],
                table,
                cutoff,
                edits,
                block_row,
                block_col,
                apply_edits=index == channel,
            )
            planes.append(plane)
            if index == channel:
                selected = block
        original = rgb
        reconstructed = np.stack(planes, axis=2)
        metric_a, metric_b = original, reconstructed
    difference = np.clip(np.abs(original.astype(np.int16) - reconstructed.astype(np.int16)) * 4, 0, 255).astype(
        np.uint8
    )
    return {
        "original": original,
        "reconstructed": reconstructed,
        "difference": difference,
        "psnr": psnr(metric_a, metric_b),
        "ssim": ssim(metric_a, metric_b),
        "block": selected,
        "quality_table": table.tolist(),
        "blocks_y": original.shape[0] // 8,
        "blocks_x": original.shape[1] // 8,
        "width": int(original.shape[1]),
        "height": int(original.shape[0]),
        "grayscale": grayscale,
        "channel": 0 if grayscale else channel,
    }


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float | None:
    mse = float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return None
    return float(10.0 * np.log10((255.0**2) / mse))


def _gaussian(size: int, sigma: float) -> np.ndarray:
    axis = np.arange(size) - size // 2
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.ndim == 3:
        return float(
            np.mean(
                [
                    ssim(original[:, :, channel], reconstructed[:, :, channel])
                    for channel in range(original.shape[2])
                ]
            )
        )
    window = _gaussian(11, 1.5)
    first = original.astype(np.float64)
    second = reconstructed.astype(np.float64)
    mu_a = convolve(first, window, mode="reflect")
    mu_b = convolve(second, window, mode="reflect")
    sigma_a = convolve(first * first, window, mode="reflect") - mu_a * mu_a
    sigma_b = convolve(second * second, window, mode="reflect") - mu_b * mu_b
    sigma_ab = convolve(first * second, window, mode="reflect") - mu_a * mu_b
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / denominator))
