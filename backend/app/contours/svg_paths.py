"""Lector de SVG: trazados, formas básicas y arcos elípticos.

Cada subtrazado (un nuevo M, o una forma distinta) es un contorno propio.
No se concatenan. El comando A se parte en cúbicas de a lo sumo 90 grados,
como indica la nota de implementación del SVG, y después se muestrea.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from app.contours.resample import resample_contour

_NUMBER = re.compile(r"[+-]?(?:(?:\d+\.\d+|\.\d+|\d+)(?:[eE][+-]?\d+)?)")
_COMMAND = re.compile(r"[MmLlHhVvCcQqSsTtAaZz]")
_TRANSFORM = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")


@dataclass(frozen=True)
class VectorContour:
    points: np.ndarray
    closed: bool
    source: str


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _numbers(text: str) -> list[float]:
    return [float(match) for match in _NUMBER.findall(text or "")]


def _transform_matrix(text: str | None) -> np.ndarray:
    """Matriz 3x3. En el atributo, la función de la derecha se aplica primero."""
    matrix = np.eye(3, dtype=np.float64)
    if not text:
        return matrix
    for name, args in _TRANSFORM.findall(text):
        values = _numbers(args)
        op = np.eye(3, dtype=np.float64)
        kind = name.lower()
        if kind == "matrix" and len(values) >= 6:
            a, b, c, d, e, f = values[:6]
            op = np.array([[a, c, e], [b, d, f], [0, 0, 1]], dtype=np.float64)
        elif kind == "translate":
            tx = values[0] if values else 0.0
            ty = values[1] if len(values) > 1 else 0.0
            op[0, 2] = tx
            op[1, 2] = ty
        elif kind == "scale":
            sx = values[0] if values else 1.0
            sy = values[1] if len(values) > 1 else sx
            op[0, 0] = sx
            op[1, 1] = sy
        elif kind == "rotate":
            angle = math.radians(values[0] if values else 0.0)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rotation = np.array(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
                dtype=np.float64,
            )
            if len(values) >= 3:
                cx, cy = values[1], values[2]
                to_origin = np.eye(3)
                to_origin[0, 2] = -cx
                to_origin[1, 2] = -cy
                back = np.eye(3)
                back[0, 2] = cx
                back[1, 2] = cy
                op = back @ rotation @ to_origin
            else:
                op = rotation
        elif kind == "skewx" and values:
            op[0, 1] = math.tan(math.radians(values[0]))
        elif kind == "skewy" and values:
            op[1, 0] = math.tan(math.radians(values[0]))
        matrix = matrix @ op
    return matrix


def _apply(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = matrix @ np.array([x, y, 1.0], dtype=np.float64)
    return float(point[0]), float(point[1])


def _cubic(p0, p1, p2, p3, steps: int = 16) -> list[tuple[float, float]]:
    samples = []
    for step in range(1, steps + 1):
        t = step / steps
        u = 1.0 - t
        x = (
            u**3 * p0[0]
            + 3 * u**2 * t * p1[0]
            + 3 * u * t**2 * p2[0]
            + t**3 * p3[0]
        )
        y = (
            u**3 * p0[1]
            + 3 * u**2 * t * p1[1]
            + 3 * u * t**2 * p2[1]
            + t**3 * p3[1]
        )
        samples.append((x, y))
    return samples


def _quad(p0, p1, p2, steps: int = 12) -> list[tuple[float, float]]:
    samples = []
    for step in range(1, steps + 1):
        t = step / steps
        u = 1.0 - t
        x = u**2 * p0[0] + 2 * u * t * p1[0] + t**2 * p2[0]
        y = u**2 * p0[1] + 2 * u * t * p1[1] + t**2 * p2[1]
        samples.append((x, y))
    return samples


def _arc_segment_cubic(cx, cy, rx, ry, phi, a0, a1):
    """Una porción de elipse, de a lo sumo 90 grados, como una cúbica."""
    alpha = a1 - a0
    k = (4.0 / 3.0) * math.tan(alpha / 4.0)
    cos0, sin0 = math.cos(a0), math.sin(a0)
    cos1, sin1 = math.cos(a1), math.sin(a1)
    p0 = (cos0, sin0)
    p1 = (cos0 - k * sin0, sin0 + k * cos0)
    p2 = (cos1 + k * sin1, sin1 - k * cos1)
    p3 = (cos1, sin1)
    cos_p, sin_p = math.cos(phi), math.sin(phi)

    def mapped(u: float, v: float) -> tuple[float, float]:
        x = rx * u
        y = ry * v
        return (
            cos_p * x - sin_p * y + cx,
            sin_p * x + cos_p * y + cy,
        )

    return mapped(*p0), mapped(*p1), mapped(*p2), mapped(*p3)


def _vector_angle(ux, uy, vx, vy) -> float:
    dot = ux * vx + uy * vy
    norm = math.hypot(ux, uy) * math.hypot(vx, vy)
    if norm < 1e-18:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / norm))
    angle = math.acos(cosine)
    if ux * vy - uy * vx < 0:
        angle = -angle
    return angle


def arc_to_cubics(x1, y1, rx, ry, phi_deg, large_arc, sweep, x2, y2):
    """Convierte un comando A en cúbicas. Lista vacía si los extremos coinciden."""
    if math.hypot(x2 - x1, y2 - y1) < 1e-12:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx < 1e-12 or ry < 1e-12:
        return [((x1, y1), (x1, y1), (x2, y2), (x2, y2))]

    phi = math.radians(phi_deg) % (2.0 * math.pi)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cos_p * dx + sin_p * dy
    y1p = -sin_p * dx + cos_p * dy

    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1.0:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale

    numerator = rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2
    denominator = rx**2 * y1p**2 + ry**2 * x1p**2
    if denominator < 1e-18:
        coef = 0.0
    else:
        coef = math.sqrt(max(0.0, numerator / denominator))
    if bool(large_arc) == bool(sweep):
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = coef * -ry * x1p / rx
    cx = cos_p * cxp - sin_p * cyp + (x1 + x2) / 2.0
    cy = sin_p * cxp + cos_p * cyp + (y1 + y2) / 2.0

    theta1 = _vector_angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    delta = _vector_angle(
        (x1p - cxp) / rx,
        (y1p - cyp) / ry,
        (-x1p - cxp) / rx,
        (-y1p - cyp) / ry,
    )
    if not sweep and delta > 0:
        delta -= 2.0 * math.pi
    elif sweep and delta < 0:
        delta += 2.0 * math.pi

    segments = max(1, math.ceil(abs(delta) / (math.pi / 2.0)))
    cubics = []
    for index in range(segments):
        a0 = theta1 + delta * index / segments
        a1 = theta1 + delta * (index + 1) / segments
        cubics.append(_arc_segment_cubic(cx, cy, rx, ry, phi, a0, a1))
    return cubics


def _tokenize_path(d: str) -> list[tuple[str, list[float]]]:
    tokens: list[tuple[str, str | float]] = []
    cursor = 0
    while cursor < len(d):
        command = _COMMAND.match(d, cursor)
        if command:
            tokens.append(("cmd", command.group(0)))
            cursor = command.end()
            continue
        number = _NUMBER.match(d, cursor)
        if number:
            tokens.append(("num", float(number.group(0))))
            cursor = number.end()
            continue
        cursor += 1
    arity = {
        "M": 2,
        "L": 2,
        "H": 1,
        "V": 1,
        "C": 6,
        "Q": 4,
        "S": 4,
        "T": 2,
        "A": 7,
        "Z": 0,
    }
    commands: list[tuple[str, list[float]]] = []
    index = 0
    while index < len(tokens):
        kind, value = tokens[index]
        if kind != "cmd":
            index += 1
            continue
        command = str(value)
        needed = arity[command.upper()]
        index += 1
        numbers: list[float] = []
        while index < len(tokens) and tokens[index][0] == "num":
            numbers.append(float(tokens[index][1]))
            index += 1
        if needed == 0:
            commands.append((command, []))
            continue
        if command.upper() == "M":
            if len(numbers) < 2:
                continue
            commands.append((command, numbers[:2]))
            rest = numbers[2:]
            implicit = "L" if command == "M" else "l"
            for offset in range(0, len(rest) - 1, 2):
                commands.append((implicit, rest[offset : offset + 2]))
            continue
        for offset in range(0, len(numbers) - needed + 1, needed):
            commands.append((command, numbers[offset : offset + needed]))
    return commands


def _sample_path(d: str, matrix: np.ndarray) -> list[VectorContour]:
    contours: list[VectorContour] = []
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    previous_cubic: tuple[float, float] | None = None
    previous_quad: tuple[float, float] | None = None
    points: list[tuple[float, float]] = []
    closed = False

    def flush() -> None:
        nonlocal points, closed
        if len(points) >= 2:
            array = np.asarray(points, dtype=np.float64)
            contours.append(VectorContour(points=array, closed=closed, source="path"))
        points = []
        closed = False

    def add_line(target: tuple[float, float]) -> None:
        nonlocal current, previous_cubic, previous_quad
        if not points:
            points.append(current)
        points.append(target)
        current = target
        previous_cubic = None
        previous_quad = None

    def add_samples(samples: list[tuple[float, float]], control_cubic=None, control_quad=None) -> None:
        nonlocal current, previous_cubic, previous_quad
        if not points:
            points.append(current)
        points.extend(samples)
        if samples:
            current = samples[-1]
        previous_cubic = control_cubic
        previous_quad = control_quad

    for command, numbers in _tokenize_path(d):
        absolute = command.isupper()
        kind = command.upper()
        if kind == "M":
            flush()
            x, y = numbers
            if not absolute:
                x += current[0]
                y += current[1]
            current = (x, y)
            start = current
            previous_cubic = None
            previous_quad = None
        elif kind == "Z":
            if points and (abs(current[0] - start[0]) > 1e-9 or abs(current[1] - start[1]) > 1e-9):
                points.append(start)
            current = start
            closed = True
            flush()
        elif kind == "L":
            x, y = numbers
            if not absolute:
                x += current[0]
                y += current[1]
            add_line((x, y))
        elif kind == "H":
            x = numbers[0]
            if not absolute:
                x += current[0]
            add_line((x, current[1]))
        elif kind == "V":
            y = numbers[0]
            if not absolute:
                y += current[1]
            add_line((current[0], y))
        elif kind == "C":
            x1, y1, x2, y2, x, y = numbers
            if not absolute:
                x1 += current[0]
                y1 += current[1]
                x2 += current[0]
                y2 += current[1]
                x += current[0]
                y += current[1]
            samples = _cubic(current, (x1, y1), (x2, y2), (x, y))
            add_samples(samples, control_cubic=(x2, y2))
        elif kind == "S":
            x2, y2, x, y = numbers
            if not absolute:
                x2 += current[0]
                y2 += current[1]
                x += current[0]
                y += current[1]
            if previous_cubic is None:
                x1, y1 = current
            else:
                x1 = 2 * current[0] - previous_cubic[0]
                y1 = 2 * current[1] - previous_cubic[1]
            samples = _cubic(current, (x1, y1), (x2, y2), (x, y))
            add_samples(samples, control_cubic=(x2, y2))
        elif kind == "Q":
            x1, y1, x, y = numbers
            if not absolute:
                x1 += current[0]
                y1 += current[1]
                x += current[0]
                y += current[1]
            samples = _quad(current, (x1, y1), (x, y))
            add_samples(samples, control_quad=(x1, y1))
        elif kind == "T":
            x, y = numbers
            if not absolute:
                x += current[0]
                y += current[1]
            if previous_quad is None:
                x1, y1 = current
            else:
                x1 = 2 * current[0] - previous_quad[0]
                y1 = 2 * current[1] - previous_quad[1]
            samples = _quad(current, (x1, y1), (x, y))
            add_samples(samples, control_quad=(x1, y1))
        elif kind == "A":
            rx, ry, phi, large, sweep, x, y = numbers
            if not absolute:
                x += current[0]
                y += current[1]
            cubics = arc_to_cubics(
                current[0],
                current[1],
                rx,
                ry,
                phi,
                large > 0.5,
                sweep > 0.5,
                x,
                y,
            )
            samples: list[tuple[float, float]] = []
            pen = current
            for p0, p1, p2, p3 in cubics:
                samples.extend(_cubic(pen, p1, p2, p3, steps=12))
                pen = p3
            if not samples:
                samples = [(x, y)]
            add_samples(samples)
    flush()

    transformed: list[VectorContour] = []
    for contour in contours:
        mapped = np.array([_apply(matrix, x, y) for x, y in contour.points], dtype=np.float64)
        transformed.append(VectorContour(points=mapped, closed=contour.closed, source=contour.source))
    return transformed


def _shape_contours(element: ET.Element, matrix: np.ndarray) -> list[VectorContour]:
    tag = _local(element.tag)
    attr = element.attrib
    if tag == "path" and attr.get("d"):
        return _sample_path(attr["d"], matrix)
    if tag == "polygon":
        values = _numbers(attr.get("points", ""))
        if len(values) < 6:
            return []
        raw = [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]
        raw.append(raw[0])
        return _sample_path("M " + " L ".join(f"{x} {y}" for x, y in raw) + " Z", matrix)
    if tag == "polyline":
        values = _numbers(attr.get("points", ""))
        if len(values) < 4:
            return []
        raw = [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]
        return _sample_path("M " + " L ".join(f"{x} {y}" for x, y in raw), matrix)
    if tag == "rect":
        x = float(attr.get("x", 0))
        y = float(attr.get("y", 0))
        w = float(attr.get("width", 0))
        h = float(attr.get("height", 0))
        if w <= 0 or h <= 0:
            return []
        d = f"M {x} {y} H {x + w} V {y + h} H {x} Z"
        return _sample_path(d, matrix)
    if tag == "circle":
        cx = float(attr.get("cx", 0))
        cy = float(attr.get("cy", 0))
        r = float(attr.get("r", 0))
        if r <= 0:
            return []
        d = (
            f"M {cx - r} {cy} "
            f"A {r} {r} 0 1 1 {cx + r} {cy} "
            f"A {r} {r} 0 1 1 {cx - r} {cy} Z"
        )
        return _sample_path(d, matrix)
    if tag == "ellipse":
        cx = float(attr.get("cx", 0))
        cy = float(attr.get("cy", 0))
        rx = float(attr.get("rx", 0))
        ry = float(attr.get("ry", 0))
        if rx <= 0 or ry <= 0:
            return []
        d = (
            f"M {cx - rx} {cy} "
            f"A {rx} {ry} 0 1 1 {cx + rx} {cy} "
            f"A {rx} {ry} 0 1 1 {cx - rx} {cy} Z"
        )
        return _sample_path(d, matrix)
    if tag == "line":
        x1 = float(attr.get("x1", 0))
        y1 = float(attr.get("y1", 0))
        x2 = float(attr.get("x2", 0))
        y2 = float(attr.get("y2", 0))
        return _sample_path(f"M {x1} {y1} L {x2} {y2}", matrix)
    return []


def _hidden(element: ET.Element) -> bool:
    if element.attrib.get("display") == "none":
        return True
    style = element.attrib.get("style", "")
    return "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", "")


def extract_svg_contours(svg_text: str, sample_count: int) -> tuple[list[dict], list[str]]:
    """Devuelve contornos remuestreados y advertencias, sin unir trazos."""
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError(f"El SVG no se pudo leer: {exc}") from exc

    found: list[VectorContour] = []

    def walk(element: ET.Element, parent: np.ndarray) -> None:
        if _hidden(element):
            return
        local = _transform_matrix(element.attrib.get("transform"))
        matrix = parent @ local
        found.extend(_shape_contours(element, matrix))
        for child in list(element):
            walk(child, matrix)

    walk(root, np.eye(3))
    warnings: list[str] = []
    if not found:
        warnings.append("El SVG no tiene trazados ni formas reconocibles.")
        return [], warnings

    contours: list[dict] = []
    open_count = 0
    for index, contour in enumerate(found):
        try:
            sampled = resample_contour(contour.points, sample_count, contour.closed)
        except ValueError:
            continue
        if not contour.closed:
            open_count += 1
        span = sampled.max(axis=0) - sampled.min(axis=0)
        contours.append(
            {
                "id": index,
                "closed": contour.closed,
                "is_hole": False,
                "perimeter": float(_polyline_length(sampled, contour.closed)),
                "sample_count": int(len(sampled)),
                "points": sampled.tolist(),
                "bbox": [
                    float(sampled[:, 0].min()),
                    float(sampled[:, 1].min()),
                    float(sampled[:, 0].max()),
                    float(sampled[:, 1].max()),
                ],
                "label": "trazado abierto" if not contour.closed else "trazado cerrado",
            }
        )
        _ = span
    if open_count:
        warnings.append(
            f"{open_count} trazado(s) abierto(s). La serie es periódica: la reconstrucción "
            "cierra el salto entre el extremo y el origen, y ahí aparece el fenómeno de Gibbs."
        )
    if len(contours) > 1:
        warnings.append(
            "Cada subtrazado es un contorno independiente. No se concatenan curvas desconectadas."
        )
    for position, contour in enumerate(contours):
        contour["id"] = position
    return contours, warnings


def _polyline_length(points: np.ndarray, closed: bool) -> float:
    if len(points) < 2:
        return 0.0
    total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if closed:
        total += float(np.linalg.norm(points[0] - points[-1]))
    return total
