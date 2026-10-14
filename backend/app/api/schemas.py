"""Modelos de la API. Los textos que ve la interfaz están en español."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PointList(BaseModel):
    points: list[tuple[float, float]] = Field(min_length=4)
    closed: bool = True
    center: bool = True
    sample_count: int = Field(default=512, ge=64, le=2048)


class PresetRequest(BaseModel):
    name: Literal["circle", "square", "star", "heart", "hummingbird"]
    sample_count: int = Field(default=512, ge=64, le=2048)


class StrokeRequest(BaseModel):
    points: list[tuple[float, float]] = Field(min_length=4)
    closed: bool | None = None
    sample_count: int = Field(default=512, ge=64, le=2048)


class FourierRequest(BaseModel):
    points: list[tuple[float, float]] = Field(min_length=4)
    closed: bool = True
    center: bool = True


class CompressionRequest(BaseModel):
    points: list[tuple[float, float]] = Field(min_length=4)
    closed: bool = True
    center: bool = True
    strategy: Literal["symmetric", "magnitude"] = "symmetric"
    component_count: int = Field(default=21, ge=1, le=500)
    number_format: Literal["float64", "float32", "int16"] = "float32"
    quant_step: float = Field(default=0.05, gt=0)


class Appearance(BaseModel):
    background: str = "#0c0e12"
    original: str = "#a9b6c6"
    reconstruction: str = "#8ef0d0"
    circles: str = "#e4b15a"
    stroke_width: float = Field(default=1.6, ge=0.5, le=8)
    show_circles: bool = True
    show_vectors: bool = True
    show_guides: bool = True
    show_original: bool = True
    color_by_error: bool = False
    mode: Literal["complex", "cartesian"] = "complex"


class ExportRequest(CompressionRequest):
    appearance: Appearance = Field(default_factory=Appearance)
    frames: int = Field(default=72, ge=12, le=120)
    width: int = Field(default=640, ge=320, le=800)
    height: int = Field(default=480, ge=240, le=800)
    fps: int = Field(default=12, ge=4, le=30)


class BlockEditModel(BaseModel):
    u: int = Field(ge=0, le=7)
    v: int = Field(ge=0, le=7)
    value: float


class JpegRequest(BaseModel):
    image_id: str
    quality: int = Field(default=50, ge=1, le=100)
    cutoff: int = Field(default=14, ge=0, le=14)
    grayscale: bool = False
    block_row: int = Field(default=0, ge=0)
    block_col: int = Field(default=0, ge=0)
    channel: int = Field(default=0, ge=0, le=2)
    edits: list[BlockEditModel] = Field(default_factory=list)
