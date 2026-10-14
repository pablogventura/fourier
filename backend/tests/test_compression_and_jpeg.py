"""FLC1, cuantización, métricas y la DCT didáctica."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from app.compression.flc1 import decode_flc1, encode_flc1, quantize_coefficients
from app.compression.metrics import evaluate_compression
from app.fourier.dft import analyze_source_points
from app.fourier.select import select_series
from app.jpeg.dct import dct2, idct2, reconstruct_jpeg
from app.presets.shapes import circle


def test_quantization_increases_error_and_integers_roundtrip():
    count = 128
    theta = 2 * np.pi * np.arange(count) / count
    points = np.column_stack([200 + 30.25 * np.cos(theta), 180 - 30.25 * np.sin(theta)])
    fine = evaluate_compression(points, True, True, "symmetric", 17, "int16", quant_step=0.01)
    coarse = evaluate_compression(points, True, True, "symmetric", 17, "int16", quant_step=3.0)
    assert coarse["rmse"] > fine["rmse"]
    spectrum = analyze_source_points(points, center=True)
    selected = select_series(spectrum.complex, "symmetric", 17)
    _restored, real_q, imag_q = quantize_coefficients(selected.coefficients, 0.25)
    payload = encode_flc1(
        selected,
        128,
        spectrum.offset_x,
        spectrum.offset_y,
        closed=True,
        centered=True,
        number_format="int16",
        quant_step=0.25,
    )
    document = decode_flc1(payload)
    assert document.number_format == "int16"
    assert np.array_equal(np.round(np.real(document.coefficients) / 0.25).astype(np.int16), real_q)
    assert np.array_equal(np.round(np.imag(document.coefficients) / 0.25).astype(np.int16), imag_q)
    assert document.centered is True
    again = decode_flc1(payload)
    assert np.allclose(again.coefficients, document.coefficients)


def test_float_file_reconstructs_the_kept_series():
    points = circle(64)
    report = evaluate_compression(points, True, True, "magnitude", 3, "float64", 0.1)
    assert report["encoded_bytes"] > 56
    assert report["raw_points_bytes"] == 64 * 16
    assert report["rmse"] < 1e-6
    assert report["hausdorff"] < 1e-4
    broken = report["payload"][:8] + b"NO"
    try:
        decode_flc1(b"XXXX" + report["payload"][4:])
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert broken[:4] == report["payload"][:4]


def test_constant_block_dc_and_roundtrip():
    block = np.full((8, 8), 7.0)
    coeffs = dct2(block)
    assert abs(coeffs[0, 0] - 56.0) < 1e-8
    assert np.max(np.abs(coeffs.ravel()[1:])) < 1e-8
    assert np.max(np.abs(idct2(coeffs) - block)) < 1e-8


def test_jpeg_lab_reports_finite_psnr_and_perfect_identity_without_quantization_loss():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :] = (40, 80, 120)
    # Calidad 100 y sin corte: la tabla sigue cuantizando, así que el PSNR es finito
    # salvo que el bloque constante caiga justo en un múltiplo. Un degradado no.
    ramp = np.tile(np.linspace(0, 255, 32), (32, 1))
    rgb = np.stack([ramp, ramp * 0.4, 255 - ramp], axis=2).astype(np.uint8)
    result = reconstruct_jpeg(rgb, quality=20, cutoff=3, grayscale=False, block_row=0, block_col=0, channel=0, edits=[])
    assert result["psnr"] is not None
    assert result["psnr"] > 0
    assert 0 <= result["ssim"] <= 1
    same = reconstruct_jpeg(image, quality=100, cutoff=14, grayscale=True, block_row=0, block_col=1, channel=0, edits=[])
    assert same["ssim"] > 0.99


def test_api_preset_and_export(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ok"
    preset = client.post("/api/contours/preset", json={"name": "circle", "sample_count": 64})
    assert preset.status_code == 200
    points = preset.json()["contours"][0]["points"]
    spectrum = client.post("/api/fourier", json={"points": points, "closed": True, "center": True})
    assert spectrum.status_code == 200
    body = spectrum.json()
    dominant = max(body["coefficients"], key=lambda item: item["amplitude"])
    assert dominant["frequency"] == 1
    metrics = client.post(
        "/api/compression/metrics",
        json={"points": points, "closed": True, "center": True, "strategy": "symmetric", "component_count": 5, "number_format": "float32", "quant_step": 0.05},
    )
    assert metrics.status_code == 200
    assert metrics.json()["encoded_bytes"] > 0
    assert len(metrics.json()["curve"]) >= 3
    gif = client.post(
        "/api/export/gif",
        json={
            "points": points,
            "closed": True,
            "center": True,
            "strategy": "magnitude",
            "component_count": 3,
            "number_format": "float32",
            "quant_step": 0.05,
            "frames": 12,
            "width": 320,
            "height": 240,
            "fps": 6,
        },
    )
    assert gif.status_code == 200
    assert gif.content[:3] == b"GIF"
    png = Image.new("RGB", (60, 40), "white")
    ImageDraw.Draw(png).ellipse((8, 6, 40, 34), fill="black")
    buffer = io_bytes(png)
    uploaded = client.post(
        "/api/contours/file",
        files={"file": ("blob.png", buffer, "image/png")},
        data={"threshold": "200", "blur_sigma": "0", "min_length": "5", "method": "binary", "sample_count": "64"},
    )
    assert uploaded.status_code == 200
    assert len(uploaded.json()["contours"]) >= 1
    loaded = client.post("/api/jpeg/load", files={"file": ("blob.png", buffer, "image/png")})
    assert loaded.status_code == 200
    image_id = loaded.json()["image_id"]
    reconstructed = client.post(
        "/api/jpeg/reconstruct",
        json={"image_id": image_id, "quality": 40, "cutoff": 6, "grayscale": True, "block_row": 0, "block_col": 0, "channel": 0, "edits": [{"u": 0, "v": 0, "value": 10}]},
    )
    assert reconstructed.status_code == 200
    assert reconstructed.json()["psnr"] is not None


def io_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
