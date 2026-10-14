"""Rutas del laboratorio. La FFT se calcula al elegir un contorno, no por fotograma."""

from __future__ import annotations

import base64
import io

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from app.api.schemas import (
    CompressionRequest,
    ExportRequest,
    FourierRequest,
    JpegRequest,
    PresetRequest,
    StrokeRequest,
)
from app.compression.flc1 import decode_flc1, encode_flc1
from app.compression.metrics import evaluate_compression, rate_distortion
from app.contours.raster import RasterParams, decode_image, extract_raster_contours, image_size
from app.contours.resample import resample_contour
from app.contours.svg_paths import extract_svg_contours
from app.export.render import encode_gif, encode_mp4, render_frames, render_still, svg_document
from app.export.tables import coefficients_csv, coefficients_json, metrics_csv, records_from_series
from app.fourier.dft import analyze_source_points, coefficient_records
from app.fourier.select import select_series
from app.jpeg.dct import BlockEdit, basis_patterns, pad_to_blocks, reconstruct_jpeg
from app.jpeg.store import IMAGES
from app.presets.shapes import build_preset, list_presets

router = APIRouter()
_BASIS = basis_patterns()


def _points(pairs: list[tuple[float, float]]) -> np.ndarray:
    array = np.asarray(pairs, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise HTTPException(status_code=400, detail="Los puntos tienen que ser pares (x, y).")
    if not np.isfinite(array).all():
        raise HTTPException(status_code=400, detail="Hay coordenadas que no son finitas.")
    return array


def _contour_dict(points: np.ndarray, closed: bool, label: str, is_hole: bool = False, note: str | None = None) -> dict:
    return {
        "id": 0,
        "closed": closed,
        "is_hole": is_hole,
        "perimeter": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() + (np.linalg.norm(points[0] - points[-1]) if closed else 0)),
        "sample_count": int(len(points)),
        "points": points.tolist(),
        "bbox": [
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        ],
        "label": label,
        "note": note,
    }


def _pack_contours(contours: list[dict], warnings: list[str], source: str, width: float, height: float, preview: str | None) -> dict:
    return {
        "source": source,
        "width": width,
        "height": height,
        "preview": preview,
        "warnings": warnings,
        "contours": contours,
    }


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/presets")
def presets() -> dict:
    return {"presets": list_presets()}


@router.post("/contours/preset")
def contour_preset(body: PresetRequest) -> dict:
    try:
        points, closed, note = build_preset(body.name, body.sample_count)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    contour = _contour_dict(points, closed, body.name, note=note)
    width = float(points[:, 0].max() - points[:, 0].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    warnings = [
        "Una serie describe un solo contorno. Si la figura tiene huecos o partes separadas, cada una es otra señal."
    ]
    if note:
        warnings.append(note)
    return _pack_contours([contour], warnings, "preset", width, height, None)


@router.post("/contours/stroke")
def contour_stroke(body: StrokeRequest) -> dict:
    if len(body.points) > 20000:
        raise HTTPException(status_code=400, detail="El trazo tiene demasiados puntos.")
    raw = _points(body.points)
    closed = body.closed
    if closed is None:
        span = np.maximum(raw.max(axis=0) - raw.min(axis=0), 1.0)
        tolerance = max(12.0, 0.04 * float(np.max(span)))
        closed = bool(np.linalg.norm(raw[0] - raw[-1]) <= tolerance)
    try:
        sampled = resample_contour(raw, body.sample_count, closed=closed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    warnings = []
    if not closed:
        warnings.append(
            "El trazo está abierto. La reconstrucción es periódica y cierra el salto: ahí aparece Gibbs."
        )
    warnings.append("El trazo se remuestreó con arcos iguales. El parámetro no es el tiempo del dibujo.")
    return _pack_contours(
        [_contour_dict(sampled, closed, "trazo")],
        warnings,
        "stroke",
        float(sampled[:, 0].max() - sampled[:, 0].min()),
        float(sampled[:, 1].max() - sampled[:, 1].min()),
        None,
    )


@router.post("/contours/file")
async def contour_file(
    file: UploadFile = File(...),
    threshold: int = Form(128),
    blur_sigma: float = Form(1.0),
    min_length: float = Form(24.0),
    method: str = Form("binary"),
    polarity: str = Form("auto"),
    sample_count: int = Form(512),
) -> dict:
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="El archivo supera los 8 MB.")
    name = (file.filename or "").lower()
    sample_count = int(np.clip(sample_count, 64, 2048))
    if _looks_like_svg(name, data):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="El SVG no está en UTF-8.") from exc
        try:
            contours, warnings = extract_svg_contours(text, sample_count)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if contours:
            stacked = np.asarray(contours[0]["points"], dtype=np.float64)
            width = float(stacked[:, 0].max() - stacked[:, 0].min())
            height = float(stacked[:, 1].max() - stacked[:, 1].min())
        else:
            width, height = 0.0, 0.0
        warnings.insert(
            0,
            "Un path con varios M se parte. Los arcos A se convierten a Bézier antes de muestrear.",
        )
        return _pack_contours(contours, warnings, "svg", width, height, None)
    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    params = RasterParams(
        threshold=int(np.clip(threshold, 0, 255)),
        blur_sigma=float(np.clip(blur_sigma, 0, 12)),
        min_length=float(max(0.0, min_length)),
        method=method if method in {"binary", "canny"} else "binary",
        polarity=polarity if polarity in {"auto", "bright", "dark"} else "auto",
        sample_count=sample_count,
    )
    contours, warnings, preview = extract_raster_contours(image, params)
    width, height = image_size(image)
    return _pack_contours(contours, warnings, "raster", float(width), float(height), preview)


def _looks_like_svg(name: str, data: bytes) -> bool:
    if name.endswith(".svg"):
        return True
    head = data[:2048].lstrip().lower()
    return b"<svg" in head


@router.post("/fourier")
def fourier(body: FourierRequest) -> dict:
    points = _points(body.points)
    if len(points) > 2048:
        raise HTTPException(status_code=400, detail="Como máximo 2048 muestras. Remuestreá el contorno.")
    spectrum = analyze_source_points(points, center=body.center)
    warnings = []
    if not body.closed:
        warnings.append(
            "El contorno está abierto. Igual se trata como un período, así que la curva reconstruida cierra el hueco."
        )
    if body.center:
        warnings.append("Centrado activo: el término de continua es la posición y los demás coeficientes describen la forma.")
    else:
        warnings.append("Centrado apagado: el primer círculo, de frecuencia 0, lleva la traslación.")
    return {
        "sample_count": spectrum.sample_count,
        "closed": body.closed,
        "centered": spectrum.centered,
        "offset": [spectrum.offset_x, spectrum.offset_y],
        "coefficients": coefficient_records(spectrum.complex),
        "x_coefficients": coefficient_records(spectrum.x),
        "y_coefficients": coefficient_records(spectrum.y),
        "warnings": warnings,
    }


def _public_metrics(report: dict) -> dict:
    return {
        key: report[key]
        for key in (
            "sample_count",
            "component_count",
            "stored_coefficients",
            "number_format",
            "quant_step",
            "raw_points_bytes",
            "raw_coefficients_bytes",
            "encoded_bytes",
            "ratio_vs_points",
            "ratio_vs_coefficients",
            "rmse",
            "max_error",
            "hausdorff",
        )
    }


@router.post("/compression/metrics")
def compression_metrics(body: CompressionRequest) -> dict:
    points = _points(body.points)
    report = evaluate_compression(
        points,
        closed=body.closed,
        center=body.center,
        strategy=body.strategy,
        component_count=body.component_count,
        number_format=body.number_format,
        quant_step=body.quant_step,
    )
    curve = rate_distortion(
        points,
        closed=body.closed,
        center=body.center,
        strategy=body.strategy,
        number_format=body.number_format,
        quant_step=body.quant_step,
        focus_count=body.component_count,
    )
    payload = _public_metrics(report)
    payload["curve"] = curve
    payload["note"] = (
        "raw_points_bytes y raw_coefficients_bytes son datos crudos en float64, sin archivo. "
        "encoded_bytes es el FLC1 real. El cociente no es una tasa JPEG, y el contorno no es la imagen."
    )
    return payload


@router.post("/compression/flc1")
def compression_file(body: CompressionRequest) -> Response:
    points = _points(body.points)
    report = evaluate_compression(
        points,
        closed=body.closed,
        center=body.center,
        strategy=body.strategy,
        component_count=body.component_count,
        number_format=body.number_format,
        quant_step=body.quant_step,
    )
    return Response(
        content=report["payload"],
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="contorno.flc1"'},
    )


@router.post("/compression/import")
async def compression_import(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    try:
        document = decode_flc1(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    points = document.reconstruct()
    records = records_from_series(document.frequencies, document.coefficients)
    return {
        "sample_count": document.sample_count,
        "closed": document.closed,
        "centered": document.centered,
        "offset": [document.offset_x, document.offset_y],
        "number_format": document.number_format,
        "quant_step": document.quant_step,
        "points": points.tolist(),
        "coefficients": records,
        "note": (
            "Estos puntos salen de los coeficientes guardados, no del contorno original. "
            "El archivo no conserva el mapa de bits ni las muestras que se descartaron."
        ),
    }


def _selected_bundle(body: CompressionRequest):
    points = _points(body.points)
    spectrum = analyze_source_points(points, center=body.center)
    selected = select_series(spectrum.complex, body.strategy, body.component_count)
    reconstructed = encode_flc1(
        selected,
        spectrum.sample_count,
        spectrum.offset_x,
        spectrum.offset_y,
        closed=body.closed,
        centered=body.center,
        number_format=body.number_format,
        quant_step=body.quant_step,
    )
    document = decode_flc1(reconstructed)
    curve = document.reconstruct()
    return points, spectrum, selected, curve


@router.post("/export/svg")
def export_svg(body: CompressionRequest) -> Response:
    points, _spectrum, _selected, curve = _selected_bundle(body)
    document = svg_document(points, curve, body.closed, "#a9b6c6", "#8ef0d0")
    return Response(
        content=document.encode("utf-8"),
        media_type="image/svg+xml",
        headers={"Content-Disposition": 'attachment; filename="reconstruccion.svg"'},
    )


@router.post("/export/png")
def export_png(body: ExportRequest) -> Response:
    points, _spectrum, _selected, curve = _selected_bundle(body)
    image = render_still(
        points,
        curve,
        body.closed,
        body.width,
        body.height,
        body.appearance.background,
        body.appearance.original,
        body.appearance.reconstruction,
        body.appearance.stroke_width,
        body.appearance.color_by_error,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="reconstruccion.png"'},
    )


def _animation(body: ExportRequest) -> list[Image.Image]:
    points = _points(body.points)
    look = body.appearance
    try:
        return render_frames(
            points,
            closed=body.closed,
            center=body.center,
            strategy=body.strategy,
            component_count=body.component_count,
            frames=body.frames,
            width=body.width,
            height=body.height,
            background=look.background,
            original_color=look.original,
            reconstruction_color=look.reconstruction,
            circle_color=look.circles,
            stroke_width=look.stroke_width,
            show_circles=look.show_circles,
            show_vectors=look.show_vectors,
            show_guides=look.show_guides,
            show_original=look.show_original,
            color_by_error=look.color_by_error,
            mode=look.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/export/gif")
def export_gif(body: ExportRequest) -> Response:
    frames = _animation(body)
    try:
        payload = encode_gif(frames, body.fps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo armar el GIF: {exc}") from exc
    return Response(
        content=payload,
        media_type="image/gif",
        headers={"Content-Disposition": 'attachment; filename="epiciclos.gif"'},
    )


@router.post("/export/mp4")
def export_mp4(body: ExportRequest) -> Response:
    frames = _animation(body)
    try:
        payload = encode_mp4(frames, body.fps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo armar el MP4: {exc}") from exc
    return Response(
        content=payload,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="epiciclos.mp4"'},
    )


@router.post("/export/coefficients.csv")
def export_coefficients_csv(body: FourierRequest) -> Response:
    points = _points(body.points)
    spectrum = analyze_source_points(points, center=body.center)
    text = coefficients_csv(coefficient_records(spectrum.complex))
    return Response(
        content=text.encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="coeficientes.csv"'},
    )


@router.post("/export/coefficients.json")
def export_coefficients_json(body: FourierRequest) -> Response:
    points = _points(body.points)
    spectrum = analyze_source_points(points, center=body.center)
    text = coefficients_json(
        {
            "sample_count": spectrum.sample_count,
            "centered": spectrum.centered,
            "offset": [spectrum.offset_x, spectrum.offset_y],
            "convention": "z = x - i y, con y de imagen hacia abajo; c_k = FFT(z) / N",
            "coefficients": coefficient_records(spectrum.complex),
            "x_coefficients": coefficient_records(spectrum.x),
            "y_coefficients": coefficient_records(spectrum.y),
        }
    )
    return Response(
        content=text.encode("utf-8"),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="coeficientes.json"'},
    )


@router.post("/export/metrics.csv")
def export_metrics_csv(body: CompressionRequest) -> Response:
    points = _points(body.points)
    report = evaluate_compression(
        points,
        closed=body.closed,
        center=body.center,
        strategy=body.strategy,
        component_count=body.component_count,
        number_format=body.number_format,
        quant_step=body.quant_step,
    )
    curve = rate_distortion(
        points,
        closed=body.closed,
        center=body.center,
        strategy=body.strategy,
        number_format=body.number_format,
        quant_step=body.quant_step,
        focus_count=body.component_count,
    )
    text = metrics_csv(_public_metrics(report), curve)
    return Response(
        content=text.encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="metricas.csv"'},
    )


@router.get("/jpeg/basis")
def jpeg_basis() -> dict:
    return {
        "size": 8,
        "order": "v * 8 + u, con u horizontal y v vertical",
        "patterns": _BASIS,
        "note": "Base ortonormal de la DCT-II. Para bloques de 8 coincide en escala con la DCT del JPEG.",
    }


@router.post("/jpeg/load")
async def jpeg_load(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    name = (file.filename or "").lower()
    if _looks_like_svg(name, data):
        raise HTTPException(status_code=400, detail="El laboratorio DCT usa una imagen rasterizada, no un SVG.")
    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rgb = np.asarray(image.convert("RGB"))
    fitted = pad_to_blocks(rgb, max_side=384)
    image_id = IMAGES.put(fitted)
    return {
        "image_id": image_id,
        "width": int(fitted.shape[1]),
        "height": int(fitted.shape[0]),
        "preview": _png_data_url(fitted),
        "note": "La imagen de trabajo se recorta a un múltiplo de 8 y a 384 px de lado como máximo.",
    }


@router.post("/jpeg/reconstruct")
def jpeg_reconstruct(body: JpegRequest) -> dict:
    try:
        rgb = IMAGES.get(body.image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Esa imagen ya no está en memoria. Volvé a cargarla.") from exc
    edits = [BlockEdit(u=item.u, v=item.v, value=item.value) for item in body.edits]
    result = reconstruct_jpeg(
        rgb,
        quality=body.quality,
        cutoff=body.cutoff,
        grayscale=body.grayscale,
        block_row=body.block_row,
        block_col=body.block_col,
        channel=body.channel,
        edits=edits,
    )
    return {
        "width": result["width"],
        "height": result["height"],
        "blocks_x": result["blocks_x"],
        "blocks_y": result["blocks_y"],
        "psnr": result["psnr"],
        "ssim": result["ssim"],
        "block": result["block"],
        "quality_table": result["quality_table"],
        "grayscale": result["grayscale"],
        "channel": result["channel"],
        "original": _png_data_url(result["original"]),
        "reconstructed": _png_data_url(result["reconstructed"]),
        "difference": _png_data_url(result["difference"]),
        "note": (
            "Esto no es un archivo JPEG. Falta el paso a YCbCr, el submuestreo de color y la codificación entrópica. "
            "La diferencia está amplificada x4 para que se vea."
        ),
    }


def _png_data_url(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
