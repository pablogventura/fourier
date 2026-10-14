"""Contornos separados, arcos SVG y máscaras con alfa."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from app.contours.raster import RasterParams, decode_image, extract_raster_contours
from app.contours.svg_paths import extract_svg_contours
from app.presets.shapes import hummingbird


SVG_TWO = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40">
  <path d="M 5 20 L 25 5 L 25 35 Z M 50 20 L 80 8 L 80 32 Z"/>
</svg>
"""

SVG_OPEN = """
<svg xmlns="http://www.w3.org/2000/svg">
  <path d="M 0 0 L 10 0 L 10 10"/>
</svg>
"""

SVG_ARC = """
<svg xmlns="http://www.w3.org/2000/svg">
  <path d="M 10 30 A 20 10 0 1 1 50 30 A 20 10 0 1 1 10 30 Z"/>
</svg>
"""

SVG_GROUP = """
<svg xmlns="http://www.w3.org/2000/svg">
  <g transform="translate(5 7)">
    <rect x="0" y="0" width="10" height="4"/>
  </g>
</svg>
"""


def test_subpaths_stay_disconnected_and_open_paths_are_marked():
    contours, warnings = extract_svg_contours(SVG_TWO, 64)
    assert len(contours) == 2
    assert all(item["closed"] for item in contours)
    assert any("no se concatenan" in warning.lower() or "independiente" in warning.lower() for warning in warnings)
    opened, open_warnings = extract_svg_contours(SVG_OPEN, 32)
    assert len(opened) == 1
    assert opened[0]["closed"] is False
    assert any("abierto" in warning.lower() for warning in open_warnings)


def test_elliptical_arc_lands_on_the_ellipse():
    contours, _warnings = extract_svg_contours(SVG_ARC, 180)
    assert len(contours) == 1
    assert contours[0]["closed"] is True
    points = np.asarray(contours[0]["points"])
    cx, cy, rx, ry = 30.0, 30.0, 20.0, 10.0
    normalized = ((points[:, 0] - cx) / rx) ** 2 + ((points[:, 1] - cy) / ry) ** 2
    assert np.max(np.abs(normalized - 1)) < 0.03


def test_group_transform_is_applied():
    contours, _warnings = extract_svg_contours(SVG_GROUP, 32)
    assert len(contours) == 1
    points = np.asarray(contours[0]["points"])
    assert points[:, 0].min() == np.min(points[:, 0])
    assert abs(points[:, 0].min() - 5) < 0.2
    assert abs(points[:, 1].min() - 7) < 0.2


def test_hummingbird_is_one_closed_shape():
    points = hummingbird(128)
    assert len(points) == 128
    assert np.linalg.norm(points[0] - points[-1]) > 1


def _png_bytes(image: Image.Image, kind: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=kind)
    return buffer.getvalue()


def test_two_blobs_are_not_merged_and_alpha_is_used():
    image = Image.new("RGB", (180, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 20, 50, 60), fill="black")
    draw.ellipse((110, 15, 160, 75), fill="black")
    decoded = decode_image(_png_bytes(image))
    contours, _warnings, _preview = extract_raster_contours(
        decoded, RasterParams(threshold=200, blur_sigma=0, min_length=10, sample_count=64)
    )
    assert len(contours) == 2
    assert all(item["closed"] for item in contours)
    perimeters = sorted(item["perimeter"] for item in contours)
    assert perimeters[0] > 40
    assert perimeters[1] < perimeters[0] + perimeters[0]

    transparent = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    ImageDraw.Draw(transparent).ellipse((15, 15, 65, 65), fill=(20, 40, 200, 255))
    decoded_alpha = decode_image(_png_bytes(transparent))
    alpha_contours, warnings, _preview = extract_raster_contours(
        decoded_alpha, RasterParams(threshold=10, blur_sigma=0, min_length=8, sample_count=64)
    )
    assert len(alpha_contours) == 1
    assert any("alfa" in warning.lower() for warning in warnings)


def test_jpeg_and_webp_decode():
    image = Image.new("RGB", (40, 30), "white")
    ImageDraw.Draw(image).rectangle((8, 8, 28, 22), fill="black")
    decode_image(_png_bytes(image, "JPEG"))
    decode_image(_png_bytes(image, "WEBP"))
