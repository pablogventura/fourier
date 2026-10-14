"""Dibuja la misma cadena de epiciclos que el canvas, para GIF y MP4.

Los fotogramas salen de coeficientes ya calculados. La FFT no se repite
en cada cuadro: solo avanza t.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from app.fourier.dft import HarmonicSeries, analyze_source_points
from app.fourier.phasors import source_chain
from app.fourier.reconstruct import reconstruct_real, reconstruct_source
from app.fourier.select import select_series


def parse_hex(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise ValueError(f"Color inválido: {value}")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))


def _fit(points: np.ndarray, width: int, height: int, pad: float) -> tuple[float, float, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    drawn_w = span[0] * scale
    drawn_h = span[1] * scale
    origin_x = (width - drawn_w) / 2.0 - minimum[0] * scale
    origin_y = (height - drawn_h) / 2.0 - minimum[1] * scale
    return float(scale), float(origin_x), float(origin_y)


def _map(points: np.ndarray, scale: float, origin_x: float, origin_y: float) -> list[tuple[float, float]]:
    screen_x = origin_x + points[:, 0] * scale
    screen_y = origin_y + points[:, 1] * scale
    return list(zip(screen_x.tolist(), screen_y.tolist(), strict=True))


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    u = max(0.0, min(1.0, t))
    return tuple(int(round(a[index] + (b[index] - a[index]) * u)) for index in range(3))


def _error_colors(original: np.ndarray, reconstructed: np.ndarray) -> list[tuple[int, int, int]]:
    low = (42, 111, 219)
    high = (232, 93, 76)
    delta = np.linalg.norm(original - reconstructed, axis=1)
    peak = float(delta.max()) if len(delta) else 1.0
    if peak < 1e-9:
        peak = 1.0
    return [_lerp(low, high, float(value / peak)) for value in delta]


def _stroke(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color, width: int, closed: bool) -> None:
    if len(pts) < 2:
        return
    sequence = pts + ([pts[0]] if closed else [])
    draw.line(sequence, fill=color, width=max(1, width), joint="curve")


def _chain_points(series: HarmonicSeries, t: float, anchor_x: float, anchor_y: float, kind: str) -> list[dict]:
    theta = np.angle(series.coefficients) + 2.0 * np.pi * series.frequencies.astype(np.float64) * t
    amplitude = np.abs(series.coefficients)
    if kind == "x":
        dx = amplitude * np.cos(theta)
        dy = -amplitude * np.sin(theta)
    else:
        dx = amplitude * np.sin(theta)
        dy = amplitude * np.cos(theta)
    cursor_x, cursor_y = anchor_x, anchor_y
    circles = []
    for index in range(len(amplitude)):
        nxt_x = cursor_x + float(dx[index])
        nxt_y = cursor_y + float(dy[index])
        circles.append(
            {
                "cx": cursor_x,
                "cy": cursor_y,
                "x": nxt_x,
                "y": nxt_y,
                "radius": float(amplitude[index]),
            }
        )
        cursor_x, cursor_y = nxt_x, nxt_y
    return circles


def render_frames(
    points: np.ndarray,
    closed: bool,
    center: bool,
    strategy: str,
    component_count: int,
    frames: int,
    width: int,
    height: int,
    background: str,
    original_color: str,
    reconstruction_color: str,
    circle_color: str,
    stroke_width: float,
    show_circles: bool,
    show_vectors: bool,
    show_guides: bool,
    show_original: bool,
    color_by_error: bool,
    mode: str,
) -> list[Image.Image]:
    spectrum = analyze_source_points(points, center=center)
    selected = select_series(spectrum.complex, strategy, component_count)
    x_selected = select_series(spectrum.x, strategy, component_count)
    y_selected = select_series(spectrum.y, strategy, component_count)
    original = np.asarray(points, dtype=np.float64)
    reconstructed = reconstruct_source(
        selected, spectrum.sample_count, spectrum.offset_x, spectrum.offset_y
    )
    colors = _error_colors(original, reconstructed) if color_by_error else None
    minimum = original.min(axis=0)
    maximum = original.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    pad_world = 0.55 * float(np.max(span))
    extra = np.vstack([minimum - pad_world, maximum + pad_world])
    bounds = np.vstack([original, reconstructed, extra])
    scale, origin_x, origin_y = _fit(bounds, width, height, pad=28)
    bg = parse_hex(background)
    ink_original = parse_hex(original_color)
    ink_curve = parse_hex(reconstruction_color)
    ink_circle = parse_hex(circle_color)
    width_px = max(1, int(round(stroke_width)))
    times = np.linspace(0.0, 1.0, frames, endpoint=False)
    pen = reconstruct_source(
        selected,
        frames,
        spectrum.offset_x,
        spectrum.offset_y,
    ) if mode == "complex" else None
    if mode != "complex":
        pen_x = reconstruct_real(x_selected, times, spectrum.offset_x)
        pen_y = reconstruct_real(y_selected, times, spectrum.offset_y)
        pen = np.column_stack([pen_x, pen_y])
    original_screen = _map(original, scale, origin_x, origin_y)
    recon_screen = _map(reconstructed, scale, origin_x, origin_y)
    y_anchor = float(minimum[1] - 0.42 * span[1])
    x_anchor = float(minimum[0] - 0.42 * span[0])
    images: list[Image.Image] = []
    for frame_index, t in enumerate(times):
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        if show_original:
            _stroke(draw, original_screen, ink_original, width_px, closed)
        if colors is None:
            _stroke(draw, recon_screen, tuple(int(c * 0.45) for c in ink_curve), width_px, closed)
        else:
            for index in range(len(recon_screen) - (0 if closed else 1)):
                a = recon_screen[index]
                b = recon_screen[(index + 1) % len(recon_screen)]
                draw.line([a, b], fill=colors[index], width=width_px)
        trail = _map(pen[: frame_index + 1], scale, origin_x, origin_y)
        _stroke(draw, trail, ink_curve, width_px + 1, closed=False)
        if mode == "complex":
            circles = source_chain(selected, float(t), spectrum.offset_x, spectrum.offset_y)
            _draw_circles(draw, circles, scale, origin_x, origin_y, ink_circle, show_circles, show_vectors)
        else:
            x_circles = _chain_points(x_selected, float(t), spectrum.offset_x, y_anchor, "x")
            y_circles = _chain_points(y_selected, float(t), x_anchor, spectrum.offset_y, "y")
            _draw_circles(draw, x_circles, scale, origin_x, origin_y, ink_circle, show_circles, show_vectors)
            _draw_circles(draw, y_circles, scale, origin_x, origin_y, ink_circle, show_circles, show_vectors)
            if show_guides and len(pen):
                px, py = pen[frame_index]
                sx, sy = origin_x + px * scale, origin_y + py * scale
                if x_circles:
                    tip_y = origin_y + x_circles[-1]["y"] * scale
                    draw.line([(sx, tip_y), (sx, sy)], fill=ink_circle, width=1)
                if y_circles:
                    tip_x = origin_x + y_circles[-1]["x"] * scale
                    draw.line([(tip_x, sy), (sx, sy)], fill=ink_circle, width=1)
        if len(pen):
            px, py = pen[frame_index]
            sx, sy = origin_x + float(px) * scale, origin_y + float(py) * scale
            r = 4
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=ink_curve)
        images.append(image)
    return images


def _draw_circles(draw, circles, scale, origin_x, origin_y, color, show_circles, show_vectors) -> None:
    for circle in circles:
        cx = origin_x + circle["cx"] * scale
        cy = origin_y + circle["cy"] * scale
        tx = origin_x + circle["x"] * scale
        ty = origin_y + circle["y"] * scale
        radius = circle["radius"] * scale
        if show_circles and radius >= 0.6:
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=color, width=1)
        if show_vectors:
            draw.line([(cx, cy), (tx, ty)], fill=color, width=2)


def render_still(
    points: np.ndarray,
    reconstructed: np.ndarray,
    closed: bool,
    width: int,
    height: int,
    background: str,
    original_color: str,
    reconstruction_color: str,
    stroke_width: float,
    color_by_error: bool,
) -> Image.Image:
    original = np.asarray(points, dtype=np.float64)
    curve = np.asarray(reconstructed, dtype=np.float64)
    bounds = np.vstack([original, curve])
    scale, origin_x, origin_y = _fit(bounds, width, height, pad=32)
    image = Image.new("RGBA", (width, height), (*parse_hex(background), 255))
    draw = ImageDraw.Draw(image)
    width_px = max(1, int(round(stroke_width)))
    _stroke(draw, _map(original, scale, origin_x, origin_y), parse_hex(original_color), width_px, closed)
    screen = _map(curve, scale, origin_x, origin_y)
    if color_by_error:
        colors = _error_colors(original, curve)
        for index in range(len(screen) - (0 if closed else 1)):
            draw.line(
                [screen[index], screen[(index + 1) % len(screen)]],
                fill=colors[index] + (255,),
                width=width_px + 1,
            )
    else:
        _stroke(draw, screen, parse_hex(reconstruction_color), width_px + 1, closed)
    return image


def encode_gif(frames: list[Image.Image], fps: int) -> bytes:
    duration = int(round(1000 / max(1, fps)))
    palette = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for frame in frames]
    buffer = io.BytesIO()
    palette[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=palette[1:],
        duration=duration,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return buffer.getvalue()


def encode_mp4(frames: list[Image.Image], fps: int) -> bytes:
    import os
    import tempfile

    import cv2

    width, height = frames[0].size
    handle, path = tempfile.mkstemp(suffix=".mp4")
    os.close(handle)
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        os.remove(path)
        raise RuntimeError("OpenCV no pudo abrir un escritor MP4. Revisá que ffmpeg esté disponible.")
    try:
        for frame in frames:
            bgr = cv2.cvtColor(np.asarray(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()
    try:
        with open(path, "rb") as handle:
            payload = handle.read()
    finally:
        os.remove(path)
    if len(payload) < 32:
        raise RuntimeError("El MP4 quedó vacío.")
    return payload


def svg_document(original: np.ndarray, reconstructed: np.ndarray, closed: bool, original_color: str, reconstruction_color: str) -> str:
    bounds = np.vstack([original, reconstructed])
    minimum = bounds.min(axis=0)
    maximum = bounds.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    pad = 0.04 * float(max(span))
    min_x = float(minimum[0] - pad)
    min_y = float(minimum[1] - pad)
    width = float(span[0] + 2 * pad)
    height = float(span[1] + 2 * pad)

    def polyline(points: np.ndarray, color: str, width: float) -> str:
        coords = " ".join(f"{x:.4f},{y:.4f}" for x, y in points)
        if closed:
            coords += f" {points[0, 0]:.4f},{points[0, 1]:.4f}"
        return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>'

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x:.4f} {min_y:.4f} {width:.4f} {height:.4f}">\n'
        f"  {polyline(original, original_color, 1.2)}\n"
        f"  {polyline(reconstructed, reconstruction_color, 1.6)}\n"
        "</svg>\n"
    )
