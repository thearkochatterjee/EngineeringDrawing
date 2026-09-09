#!/usr/bin/env python3
r"""Render ground-truth and predicted CAD primitives over a drawing image.

Example:
  python render_primitive_overlay.py ^
    --image data\ParaCAD\data\ParaCAD\dxfs_color1_pngs\1897-2_1_0.png ^
    --prediction overlay_prediction.json ^
    --ground-truth-jsonl ParaCAD_processed_v2\train.jsonl ^
    --output overlay_comparison.jpg
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from cli_progress import RichProgress, add_progress_argument


GROUND_TRUTH_COLOR = (0, 220, 255)
PREDICTION_COLOR = (255, 64, 180)
SUPPORTED_TYPES = {"line", "circle", "arc"}


def load_ground_truth(jsonl_path: Path, image_path: Path) -> list[dict[str, Any]]:
    """Find the processed-label record corresponding to an image file."""
    image_name = image_path.name
    image_parent = image_path.parent.name
    fallback: list[dict[str, Any]] | None = None
    with jsonl_path.open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if Path(record.get("image", "")).name != image_name:
                continue
            primitives = record.get("primitives", [])
            stored_parent = Path(str(record.get("image_path", "")).replace("\\", "/")).parent.name
            if stored_parent == image_parent:
                return primitives
            fallback = primitives
    if fallback is not None:
        return fallback
    raise ValueError(f"No ground-truth record for {image_name} was found in {jsonl_path}.")


def normalized_radius_to_pixels(radius: float, width: int, height: int) -> float:
    # ParaCAD's normalized circle/arc radius is scaled by max(width, height).
    return radius * max(width, height)


def draw_primitives(
    image: Image.Image,
    primitives: Iterable[dict[str, Any]],
    color: tuple[int, int, int],
    prediction_format: bool,
) -> int:
    """Draw normalized ParaCAD primitives and return the number rendered."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    stroke_width = max(2, round(max(width, height) / 300))
    count = 0
    for primitive in primitives:
        kind = str(primitive.get("type", "")).lower()
        if kind not in SUPPORTED_TYPES:
            continue
        geometry = primitive.get("geometry", {}) or {}
        try:
            if kind == "line":
                draw.line(
                    (
                        float(geometry["x1"]) * width,
                        float(geometry["y1"]) * height,
                        float(geometry["x2"]) * width,
                        float(geometry["y2"]) * height,
                    ),
                    fill=color,
                    width=stroke_width,
                )
            else:
                cx = float(geometry["cx"]) * width
                cy = float(geometry["cy"]) * height
                radius = normalized_radius_to_pixels(float(geometry["radius"] if prediction_format else geometry["r"]), width, height)
                bounds = (cx - radius, cy - radius, cx + radius, cy + radius)
                if kind == "circle":
                    draw.ellipse(bounds, outline=color, width=stroke_width)
                else:
                    draw.arc(
                        bounds,
                        start=float(geometry["start_param"]),
                        end=float(geometry["end_param"]),
                        fill=color,
                        width=stroke_width,
                    )
            count += 1
        except (KeyError, TypeError, ValueError):
            # Skip malformed rows rather than hiding all useful comparison data.
            continue
    return count


def primitive_label(primitive: dict[str, Any], prediction_format: bool) -> str:
    if prediction_format:
        return f"P{primitive.get('query_index', '?')}"
    return str(primitive.get("id", "GT"))


def geometry_summary(primitive: dict[str, Any] | None, prediction_format: bool) -> str:
    """Format normalized geometry compactly for a human-readable diagnostic table."""
    if primitive is None:
        return "-"
    kind = str(primitive.get("type", "")).lower()
    geometry = primitive.get("geometry", {}) or {}
    try:
        if kind == "line":
            return "({:.3f}, {:.3f}) to ({:.3f}, {:.3f})".format(
                float(geometry["x1"]), float(geometry["y1"]), float(geometry["x2"]), float(geometry["y2"])
            )
        radius_key = "radius" if prediction_format else "r"
        if kind == "circle":
            return "center=({:.3f}, {:.3f}), r={:.3f}".format(
                float(geometry["cx"]), float(geometry["cy"]), float(geometry[radius_key])
            )
        if kind == "arc":
            return "center=({:.3f}, {:.3f}), r={:.3f}, {:.1f} to {:.1f} deg".format(
                float(geometry["cx"]),
                float(geometry["cy"]),
                float(geometry[radius_key]),
                float(geometry["start_param"]),
                float(geometry["end_param"]),
            )
    except (KeyError, TypeError, ValueError):
        pass
    return "unavailable"


def geometry_error(
    ground_truth: dict[str, Any], prediction: dict[str, Any]
) -> float:
    """Mean normalized coordinate error for primitives of the same type."""
    kind = str(ground_truth.get("type", "")).lower()
    if kind != str(prediction.get("type", "")).lower():
        return math.inf
    gt = ground_truth["geometry"]
    pred = prediction["geometry"]
    if kind == "line":
        values = [
            abs(float(gt["x1"]) - float(pred["x1"])),
            abs(float(gt["y1"]) - float(pred["y1"])),
            abs(float(gt["x2"]) - float(pred["x2"])),
            abs(float(gt["y2"]) - float(pred["y2"])),
        ]
    else:
        values = [
            abs(float(gt["cx"]) - float(pred["cx"])),
            abs(float(gt["cy"]) - float(pred["cy"])),
            abs(float(gt["r"]) - float(pred["radius"])),
        ]
        if kind == "arc":
            def angle_distance(a: float, b: float) -> float:
                return abs((a - b + 180.0) % 360.0 - 180.0) / 180.0

            values.extend(
                [
                    angle_distance(float(gt["start_param"]), float(pred["start_param"])),
                    angle_distance(float(gt["end_param"]), float(pred["end_param"])),
                ]
            )
    return sum(values) / len(values)


def match_primitives(
    ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    """Greedily pair same-class primitives under a transparent geometry threshold."""
    unmatched_predictions = set(range(len(predictions)))
    rows: list[dict[str, Any]] = []
    for gt in ground_truth:
        candidates = [
            (geometry_error(gt, predictions[index]), index)
            for index in unmatched_predictions
            if str(predictions[index].get("type", "")).lower() == str(gt.get("type", "")).lower()
        ]
        if candidates:
            error, prediction_index = min(candidates)
            if error <= threshold:
                unmatched_predictions.remove(prediction_index)
                rows.append(
                    {
                        "status": "matched",
                        "ground_truth": gt,
                        "prediction": predictions[prediction_index],
                        "geometry_mae": round(error, 6),
                    }
                )
                continue
        rows.append({"status": "missed_ground_truth", "ground_truth": gt, "prediction": None, "geometry_mae": None})
    for prediction_index in sorted(unmatched_predictions):
        rows.append(
            {
                "status": "extra_prediction",
                "ground_truth": None,
                "prediction": predictions[prediction_index],
                "geometry_mae": None,
            }
        )
    matched = sum(row["status"] == "matched" for row in rows)
    return {
        "match_threshold": threshold,
        "summary": {
            "ground_truth_primitives": len(ground_truth),
            "predicted_primitives": len(predictions),
            "matched": matched,
            "missed_ground_truth": sum(row["status"] == "missed_ground_truth" for row in rows),
            "extra_predictions": sum(row["status"] == "extra_prediction" for row in rows),
        },
        "comparisons": rows,
    }


def create_comparison(
    image_path: Path,
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    max_side: int,
) -> Image.Image:
    original = Image.open(image_path).convert("RGB")
    scale = min(1.0, max_side / max(original.size))
    panel_size = (round(original.width * scale), round(original.height * scale))
    original = original.resize(panel_size, Image.Resampling.LANCZOS)
    ground_truth_panel = original.copy()
    prediction_panel = original.copy()
    ground_truth_count = draw_primitives(ground_truth_panel, ground_truth, GROUND_TRUTH_COLOR, prediction_format=False)
    prediction_count = draw_primitives(prediction_panel, predictions, PREDICTION_COLOR, prediction_format=True)

    gap = 16
    header_height = 44
    output = Image.new("RGB", (panel_size[0] * 3 + gap * 2, panel_size[1] + header_height), "white")
    output.paste(original, (0, header_height))
    output.paste(ground_truth_panel, (panel_size[0] + gap, header_height))
    output.paste(prediction_panel, (panel_size[0] * 2 + gap * 2, header_height))
    draw = ImageDraw.Draw(output)
    draw.text((8, 12), "Original image", fill="black")
    draw.text((panel_size[0] + gap + 8, 12), f"Ground truth (cyan): {ground_truth_count} primitives", fill=GROUND_TRUTH_COLOR)
    draw.text((panel_size[0] * 2 + gap * 2 + 8, 12), f"Prediction (magenta): {prediction_count} primitives", fill=PREDICTION_COLOR)
    return output


def write_inline_html(image: Image.Image, destination: Path, analysis: dict[str, Any]) -> None:
    """Embed the comparison and per-primitive match details in an HTML fragment."""
    data = io.BytesIO()
    image.save(data, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(data.getvalue()).decode("ascii")
    rows: list[str] = []
    for row in analysis["comparisons"]:
        ground_truth = row["ground_truth"]
        prediction = row["prediction"]
        gt_text = primitive_label(ground_truth, False) if ground_truth else "-"
        predicted_text = primitive_label(prediction, True) if prediction else "-"
        gt_geometry = geometry_summary(ground_truth, False)
        predicted_geometry = geometry_summary(prediction, True)
        score = f"{float(prediction.get('score', 0.0)):.3f}" if prediction else "-"
        error = f"{float(row['geometry_mae']):.4f}" if row["geometry_mae"] is not None else "-"
        rows.append(
            "    <tr>"
            f"<td>{html.escape(row['status'].replace('_', ' '))}</td>"
            f"<td>{html.escape(gt_text)}</td>"
            f"<td>{html.escape(gt_geometry)}</td>"
            f"<td>{html.escape(predicted_text)}</td>"
            f"<td>{html.escape(predicted_geometry)}</td>"
            f"<td class=\"text-end\">{score}</td>"
            f"<td class=\"text-end\">{error}</td>"
            "</tr>"
        )
    summary = analysis["summary"]
    fragment = (
        '<div id="drawing-diagnostic">\n'
        f'  <img src="data:image/jpeg;base64,{encoded}" '
        'alt="Original engineering drawing, cyan ground-truth primitives, and magenta model predictions in side-by-side panels." style="max-width:100%;height:auto;">\n'
        '  <div class="viz-row">\n'
        f'    <span class="viz-badge">Matched: {summary["matched"]}</span>\n'
        f'    <span class="viz-badge">Missed ground truth: {summary["missed_ground_truth"]}</span>\n'
        f'    <span class="viz-badge">Extra predictions: {summary["extra_predictions"]}</span>\n'
        "  </div>\n"
        '  <div class="table-responsive">\n'
        '    <table class="table table-sm">\n'
        "      <thead><tr><th>Status</th><th>Ground truth</th><th>Ground-truth geometry</th><th>Prediction</th><th>Predicted geometry</th><th class=\"text-end\">Score</th><th class=\"text-end\">Geometry MAE</th></tr></thead>\n"
        f"      <tbody>\n{chr(10).join(rows)}\n      </tbody>\n"
        "    </table>\n"
        "  </div>\n"
        "</div>\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(fragment, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw ParaCAD ground truth and model predictions over an image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True, help="JSON produced by paracad_primitive_detr.py predict.")
    parser.add_argument("--ground-truth-jsonl", type=Path, required=True, help="Processed ParaCAD split JSONL containing this image.")
    parser.add_argument("--output", type=Path, default=Path("primitive_overlay.jpg"))
    parser.add_argument("--details", type=Path, default=Path("primitive_overlay_details.json"), help="Detailed match report JSON.")
    parser.add_argument("--html", type=Path, help="Optional in-conversation HTML fragment containing the comparison.")
    parser.add_argument("--max-side", type=int, default=760)
    parser.add_argument("--match-threshold", type=float, default=0.03, help="Maximum normalized geometry MAE for a match.")
    add_progress_argument(parser)
    args = parser.parse_args()
    if not args.image.is_file() or not args.prediction.is_file() or not args.ground_truth_jsonl.is_file():
        parser.error("--image, --prediction, and --ground-truth-jsonl must all be existing files.")

    with RichProgress(enabled=not args.no_progress) as progress:
        task = progress.add_task("Rendering prediction and ground-truth overlay", total=4)
        prediction_payload = json.loads(args.prediction.read_text(encoding="utf-8"))
        predictions = [item for item in prediction_payload.get("primitives", []) if item.get("type") in SUPPORTED_TYPES]
        ground_truth = [item for item in load_ground_truth(args.ground_truth_jsonl, args.image) if item.get("type") in SUPPORTED_TYPES]
        progress.advance(task)
        analysis = match_primitives(ground_truth, predictions, args.match_threshold)
        comparison = create_comparison(
            args.image,
            predictions,
            ground_truth,
            args.max_side,
        )
        progress.advance(task)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(args.output, quality=94, optimize=True)
        progress.advance(task)
        args.details.parent.mkdir(parents=True, exist_ok=True)
        args.details.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        if args.html:
            write_inline_html(comparison, args.html, analysis)
        progress.complete(task, "Overlay rendering complete")
    print(f"[DONE] Wrote overlay to {args.output} and match details to {args.details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
