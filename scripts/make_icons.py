"""Generate PWA icons (brand spark on indigo) into frontend/public via PyMuPDF."""

from __future__ import annotations

import math
from pathlib import Path

import fitz  # PyMuPDF

OUT = Path(__file__).resolve().parent.parent / "frontend" / "public"
OUT.mkdir(parents=True, exist_ok=True)

# Brand colours (0-1 float RGB)
INDIGO = (0.388, 0.400, 0.945)   # #6366f1
VIOLET = (0.545, 0.361, 0.965)   # #8b5cf6
WHITE = (1.0, 1.0, 1.0)


def spark_points(cx: float, cy: float, outer: float, inner: float) -> list[fitz.Point]:
    """8-vertex 4-point sparkle polygon."""
    pts: list[fitz.Point] = []
    for i in range(8):
        ang = math.radians(i * 45 - 90)
        r = outer if i % 2 == 0 else inner
        pts.append(fitz.Point(cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def make_icon(size: int, path: Path, spark_scale: float = 0.5) -> None:
    doc = fitz.open()
    page = doc.new_page(width=size, height=size)
    shape = page.new_shape()

    # Full-bleed background (maskable-safe). Two diagonal triangles fake a
    # gradient from indigo to violet.
    shape.draw_polyline(
        [fitz.Point(0, 0), fitz.Point(size, 0), fitz.Point(size, size)]
    )
    shape.finish(fill=VIOLET, color=None, closePath=True)
    shape.draw_polyline(
        [fitz.Point(0, 0), fitz.Point(0, size), fitz.Point(size, size)]
    )
    shape.finish(fill=INDIGO, color=None, closePath=True)

    # White spark in the centre.
    c = size / 2
    outer = size * spark_scale / 2
    inner = outer * 0.36
    shape.draw_polyline(spark_points(c, c, outer, inner))
    shape.finish(fill=WHITE, color=None, closePath=True)

    shape.commit()
    pix = page.get_pixmap(alpha=False)
    pix.save(str(path))
    print(f"wrote {path.name} ({size}x{size})")


if __name__ == "__main__":
    make_icon(192, OUT / "icon-192.png", spark_scale=0.52)
    make_icon(512, OUT / "icon-512.png", spark_scale=0.52)
    make_icon(512, OUT / "icon-maskable.png", spark_scale=0.40)  # padded safe zone
    make_icon(180, OUT / "apple-touch-icon.png", spark_scale=0.52)
