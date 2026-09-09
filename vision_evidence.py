"""Prepare compact, labeled vision evidence for Ollama-capable drawing models."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PREDICTION_COLOR = (255, 56, 170)


def _resize(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)


def _encode(image: Image.Image, max_side: int) -> str:
    buffer = io.BytesIO()
    _resize(image.convert("RGB"), max_side).save(buffer, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _draw_predictions(image: Image.Image, features: list[dict[str, Any]]) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    stroke = max(2, round(max(width, height) / 350))
    for feature in features:
        geometry = feature.get("geometry", {})
        try:
            if feature.get("type") == "line":
                draw.line(
                    (geometry["x1"] * width, geometry["y1"] * height, geometry["x2"] * width, geometry["y2"] * height),
                    fill=PREDICTION_COLOR,
                    width=stroke,
                )
            elif feature.get("type") in {"circle", "arc"}:
                center_x, center_y = geometry["cx"] * width, geometry["cy"] * height
                radius = geometry["radius"] * max(width, height)
                bounds = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
                if feature["type"] == "circle":
                    draw.ellipse(bounds, outline=PREDICTION_COLOR, width=stroke)
                else:
                    draw.arc(bounds, geometry["start_param"], geometry["end_param"], fill=PREDICTION_COLOR, width=stroke)
        except (KeyError, TypeError, ValueError):
            continue
    return overlay


def _geometry_crop_box(features: list[dict[str, Any]], width: int, height: int) -> tuple[int, int, int, int] | None:
    bounds = []
    for feature in features:
        metrics = feature.get("metrics", {})
        bbox = metrics.get("bbox_normalized") or {}
        try:
            bounds.append((float(bbox["x_min"]), float(bbox["y_min"]), float(bbox["x_max"]), float(bbox["y_max"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not bounds:
        return None
    x_min, y_min = min(row[0] for row in bounds), min(row[1] for row in bounds)
    x_max, y_max = max(row[2] for row in bounds), max(row[3] for row in bounds)
    padding = 0.04
    x_min, y_min = max(0.0, x_min - padding), max(0.0, y_min - padding)
    x_max, y_max = min(1.0, x_max + padding), min(1.0, y_max + padding)
    if x_max - x_min > 0.93 and y_max - y_min > 0.93:
        return None
    return (math.floor(x_min * width), math.floor(y_min * height), math.ceil(x_max * width), math.ceil(y_max * height))


def build_vision_evidence(
    image_path: Path, facts: dict[str, Any], mode: str = "auto", max_side: int = 1200, max_images: int = 3
) -> list[dict[str, str]]:
    """Return ordered image payloads accepted by Ollama's ``messages.images`` field.

    ``auto`` produces the original drawing, the detector overlay, and one
    focused crop. The crop is only visual context—the tool/facts record remains
    the source of stable feature IDs and numeric values.
    """
    if mode == "none":
        return []
    if not image_path.is_file():
        raise FileNotFoundError(f"Vision image not found: {image_path}")
    if max_side < 256 or max_images < 1:
        raise ValueError("--vision-max-side must be at least 256 and --vision-max-images at least 1.")
    original = Image.open(image_path).convert("RGB")
    features = list(facts.get("features", {}).get("predicted_primitives", []))
    evidence: list[tuple[str, str, Image.Image]] = [
        ("original-drawing", "Original raster engineering drawing. Do not infer physical units from pixels.", original)
    ]
    if mode == "auto":
        evidence.append(
            (
                "predicted-geometry-overlay",
                "Original drawing with detector-predicted primitives in magenta. Use facts tools to retrieve feature IDs and scores; visual overlay is verification context only.",
                _draw_predictions(original, features),
            )
        )
        crop_box = _geometry_crop_box(features, original.width, original.height)
        title_candidates = facts.get("annotations", {}).get("title_block", {}).get("candidate_text", [])
        if title_candidates:
            focus = original.crop((round(original.width * 0.55), round(original.height * 0.55), original.width, original.height))
            evidence.append(("title-block-crop", "Lower-right title-block candidate crop. OCR remains unverified.", focus))
        elif crop_box:
            evidence.append(("detected-geometry-crop", "Crop around the detected geometry extent; verify detector overlay against the original.", original.crop(crop_box)))
    return [
        {"id": identifier, "description": description, "base64_jpeg": _encode(image, max_side)}
        for identifier, description, image in evidence[:max_images]
    ]


def render_prediction_overlay(image_path: Path, facts: dict[str, Any]) -> Image.Image:
    """Return a display-ready image with detector primitives in magenta."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Overlay image not found: {image_path}")
    return _draw_predictions(Image.open(image_path).convert("RGB"), list(facts.get("features", {}).get("predicted_primitives", [])))
