"""Imágenes del laboratorio DCT, solo en memoria y con cupo chico."""

from __future__ import annotations

import uuid

import numpy as np


class ImageStore:
    def __init__(self, limit: int = 3) -> None:
        self._items: dict[str, np.ndarray] = {}
        self._order: list[str] = []
        self._limit = limit

    def put(self, rgb: np.ndarray) -> str:
        key = uuid.uuid4().hex[:12]
        self._items[key] = rgb
        self._order.append(key)
        while len(self._order) > self._limit:
            oldest = self._order.pop(0)
            self._items.pop(oldest, None)
        return key

    def get(self, key: str) -> np.ndarray:
        if key not in self._items:
            raise KeyError(key)
        return self._items[key]


IMAGES = ImageStore()
