"""Contornos de una imagen rasterizada.

OpenCV devuelve cada curva por separado. Un agujero y el borde exterior
siguen siendo dos señales: una serie compleja no puede describir su unión.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.contours.resample import perimeter, resample_contour

MAX_SIDE = 1600
MAX_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class RasterParams:
    threshold: int = 128
    blur_sigma: float = 1.0
    min_length: float = 24.0
    method: str = "binary"
    polarity: str = "auto"
    canny_low: float = 40.0
    canny_high: float = 120.0
    sample_count: int = 512


def decode_image(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise ValueError("La imagen supera los 8 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise ValueError("No se pudo leer la imagen. Usá PNG, JPG o WebP.") from exc
    image = ImageOps.exif_transpose(image)
    image = _limit_side(image, MAX_SIDE)
    return image


def _limit_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.LANCZOS,
    )
    return resized


def _has_useful_alpha(image: Image.Image) -> bool:
    if image.mode == "P" and "transparency" in image.info:
        rgba = image.convert("RGBA")
    elif "A" not in image.getbands():
        return False
    else:
        rgba = image.convert("RGBA")
    alpha = np.asarray(rgba)[:, :, 3]
    return bool(alpha.min() < 250 and alpha.max() > 5)


def working_array(image: Image.Image) -> tuple[np.ndarray, np.ndarray, bool]:
    """RGB uint8, máscara de gris uint8 y si el gris salió del alfa."""
    used_alpha = _has_useful_alpha(image)
    rgba = np.asarray(image.convert("RGBA"))
    rgb = rgba[:, :, :3].copy()
    if used_alpha:
        gray = rgba[:, :, 3]
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray, used_alpha


def extract_raster_contours(image: Image.Image, params: RasterParams) -> tuple[list[dict], list[str], str]:
    rgb, gray, used_alpha = working_array(image)
    warnings: list[str] = []
    if used_alpha:
        warnings.append("Se usó el canal alfa como máscara, así el fondo transparente no genera un rectángulo.")
    blurred = gray
    if params.blur_sigma >= 0.05:
        blurred = cv2.GaussianBlur(gray, (0, 0), float(params.blur_sigma))
    method = params.method if params.method in {"binary", "canny"} else "binary"
    if method == "canny":
        mask = cv2.Canny(blurred, float(params.canny_low), float(params.canny_high))
        warnings.append(
            "Canny marca bordes. El contorno que devuelve OpenCV recorre el borde como un lazo fino, no rellena la figura."
        )
    else:
        threshold = int(np.clip(params.threshold, 0, 255))
        _level, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        mask, polarity_note = _apply_polarity(mask, params.polarity)
        if polarity_note:
            warnings.append(polarity_note)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        warnings.append("No se detectó ningún contorno. Probá otro umbral, menos desenfoque o el método de Canny.")
        preview = _png_data_url(rgb)
        return [], warnings, preview

    extracted: list[dict] = []
    for index, contour in enumerate(contours):
        raw = contour.reshape(-1, 2).astype(np.float64)
        if len(raw) < 4:
            continue
        try:
            length = perimeter(raw, closed=True)
        except ValueError:
            continue
        if length < params.min_length:
            continue
        sampled = resample_contour(raw, params.sample_count, closed=True)
        depth = _hole_depth(hierarchy, index)
        extracted.append(
            {
                "closed": True,
                "is_hole": depth % 2 == 1,
                "perimeter": length,
                "sample_count": int(len(sampled)),
                "points": sampled.tolist(),
                "bbox": [
                    float(sampled[:, 0].min()),
                    float(sampled[:, 1].min()),
                    float(sampled[:, 0].max()),
                    float(sampled[:, 1].max()),
                ],
                "label": "agujero" if depth % 2 == 1 else "contorno",
            }
        )
    extracted.sort(key=lambda item: item["perimeter"], reverse=True)
    for index, item in enumerate(extracted):
        item["id"] = index
    if not extracted:
        warnings.append("Los contornos encontrados son más cortos que la longitud mínima.")
    elif len(extracted) > 1:
        warnings.append(
            "Hay varios contornos. Elegí uno solo: una serie de Fourier describe una curva, no la unión de piezas sueltas."
        )
    return extracted, warnings, _png_data_url(rgb)


def _apply_polarity(mask: np.ndarray, polarity: str) -> tuple[np.ndarray, str | None]:
    """`bright` deja en blanco lo que supera el umbral. `dark` lo invierte.

    `auto` invierte si el borde de la imagen quedó como figura, que es lo que
    pasa con un dibujo oscuro sobre fondo claro.
    """
    if polarity == "dark":
        return 255 - mask, "Polaridad oscura: la figura es lo que está debajo del umbral."
    if polarity == "bright":
        return mask, None
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if float(border.mean()) > 127:
        return 255 - mask, "Se invirtió la máscara: el borde de la imagen estaba del lado de la figura."
    return mask, None


def _hole_depth(hierarchy: np.ndarray, index: int) -> int:
    depth = 0
    parent = int(hierarchy[0][index][3])
    seen: set[int] = set()
    while parent >= 0 and parent not in seen:
        seen.add(parent)
        depth += 1
        parent = int(hierarchy[0][parent][3])
    return depth


def _png_data_url(rgb: np.ndarray) -> str:
    import base64

    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_size(image: Image.Image) -> tuple[int, int]:
    return image.size
