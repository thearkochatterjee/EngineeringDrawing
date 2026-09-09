"""Grounded, local facts and checks for an engineering-drawing image.

This module deliberately separates three kinds of evidence:

* ``detector`` facts produced by ``paracad_primitive_detr.py``;
* ``label`` facts read from ParaCAD JSONL, when supplied for evaluation; and
* ``ocr`` text, which is useful but never treated as verified CAD metadata.

All geometry is expressed in normalized image coordinates, with a pixel
equivalent included where useful.  Physical units cannot be inferred from a
raster drawing alone, so callers must not present image measurements as
manufacturing dimensions.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SUPPORTED_TYPES = {"line", "circle", "arc"}
SCHEMA_VERSION = "1.0"


def _number(value: Any) -> float | None:
    """Return a finite float, or None for missing/non-finite data."""
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _find_record(jsonl_path: Path, image_path: Path) -> dict[str, Any]:
    """Load the label record whose file name (and preferably parent) match."""
    fallback: dict[str, Any] | None = None
    with jsonl_path.open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if Path(str(record.get("image", ""))).name != image_path.name:
                continue
            if Path(str(record.get("image_path", "")).replace("\\", "/")).parent.name == image_path.parent.name:
                return record
            fallback = record
    if fallback is None:
        raise ValueError(f"No label record for {image_path.name!r} was found in {jsonl_path}.")
    return fallback


def _canonical_geometry(primitive: dict[str, Any]) -> dict[str, float] | None:
    """Convert detector or ParaCAD geometry to one compact representation."""
    kind = str(primitive.get("type", "")).lower()
    source = primitive.get("geometry") or {}
    if kind == "line":
        values = {key: _number(source.get(key)) for key in ("x1", "y1", "x2", "y2")}
    elif kind in {"circle", "arc"}:
        values = {
            "cx": _number(source.get("cx")),
            "cy": _number(source.get("cy")),
            "radius": _number(source.get("radius", source.get("r"))),
        }
        if kind == "arc":
            values["start_param"] = _number(source.get("start_param"))
            values["end_param"] = _number(source.get("end_param"))
    else:
        return None
    return {key: value for key, value in values.items() if value is not None} if all(value is not None for value in values.values()) else None


def _feature_id(primitive: dict[str, Any], source: str, index: int) -> str:
    if source == "detector":
        query_index = primitive.get("query_index", index)
        return f"P{query_index}"
    raw_id = str(primitive.get("id", f"{primitive.get('type', 'primitive')}{index}"))
    return f"GT-{raw_id}"


def _line_orientation(geometry: dict[str, float], tolerance: float = 0.002) -> str:
    dx = geometry["x2"] - geometry["x1"]
    dy = geometry["y2"] - geometry["y1"]
    if abs(dy) <= tolerance:
        return "horizontal"
    if abs(dx) <= tolerance:
        return "vertical"
    return "diagonal"


def _arc_sweep(start: float, end: float) -> float:
    sweep = (end - start) % 360.0
    return 360.0 if math.isclose(sweep, 0.0, abs_tol=1e-7) else sweep


def _feature_points(feature: dict[str, Any]) -> list[tuple[float, float]]:
    geometry = feature["geometry"]
    if feature["type"] == "line":
        return [(geometry["x1"], geometry["y1"]), (geometry["x2"], geometry["y2"])]
    if feature["type"] == "arc":
        start = math.radians(geometry["start_param"])
        end = math.radians(geometry["end_param"])
        return [
            (geometry["cx"] + geometry["radius"] * math.cos(start), geometry["cy"] + geometry["radius"] * math.sin(start)),
            (geometry["cx"] + geometry["radius"] * math.cos(end), geometry["cy"] + geometry["radius"] * math.sin(end)),
        ]
    return []


def feature_center(feature: dict[str, Any]) -> tuple[float, float]:
    geometry = feature["geometry"]
    if feature["type"] == "line":
        return ((geometry["x1"] + geometry["x2"]) / 2, (geometry["y1"] + geometry["y2"]) / 2)
    return (geometry["cx"], geometry["cy"])


def _bbox(feature: dict[str, Any]) -> dict[str, float]:
    geometry = feature["geometry"]
    if feature["type"] == "line":
        xs, ys = (geometry["x1"], geometry["x2"]), (geometry["y1"], geometry["y2"])
        return {"x_min": min(xs), "y_min": min(ys), "x_max": max(xs), "y_max": max(ys)}
    radius = geometry["radius"]
    return {
        "x_min": geometry["cx"] - radius,
        "y_min": geometry["cy"] - radius,
        "x_max": geometry["cx"] + radius,
        "y_max": geometry["cy"] + radius,
    }


def _shape_metrics(feature: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    geometry = feature["geometry"]
    bbox = _bbox(feature)
    center = feature_center(feature)
    metrics: dict[str, Any] = {
        "center_normalized": {"x": _round(center[0]), "y": _round(center[1])},
        "center_pixels": {"x": _round(center[0] * width, 2), "y": _round(center[1] * height, 2)},
        "bbox_normalized": {key: _round(value) for key, value in bbox.items()},
        "bbox_pixels": {
            "x_min": _round(bbox["x_min"] * width, 2),
            "y_min": _round(bbox["y_min"] * height, 2),
            "x_max": _round(bbox["x_max"] * width, 2),
            "y_max": _round(bbox["y_max"] * height, 2),
        },
    }
    if feature["type"] == "line":
        normalized_length = _distance((geometry["x1"], geometry["y1"]), (geometry["x2"], geometry["y2"]))
        metrics.update(
            {
                "orientation": _line_orientation(geometry),
                "length_normalized": _round(normalized_length),
                "length_pixels_approx": _round(
                    math.hypot((geometry["x2"] - geometry["x1"]) * width, (geometry["y2"] - geometry["y1"]) * height), 2
                ),
            }
        )
    else:
        metrics["radius_normalized"] = _round(geometry["radius"])
        metrics["radius_pixels_approx"] = _round(geometry["radius"] * max(width, height), 2)
        if feature["type"] == "arc":
            sweep = _arc_sweep(geometry["start_param"], geometry["end_param"])
            metrics["sweep_degrees"] = _round(sweep, 3)
            metrics["arc_length_normalized"] = _round(geometry["radius"] * math.radians(sweep))
    return metrics


def make_features(
    primitives: Iterable[dict[str, Any]], source: str, width: int, height: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create serializable, stable feature records and report malformed rows."""
    features: list[dict[str, Any]] = []
    warnings: list[str] = []
    used_ids: set[str] = set()
    for index, primitive in enumerate(primitives):
        kind = str(primitive.get("type", "")).lower()
        geometry = _canonical_geometry(primitive)
        if kind not in SUPPORTED_TYPES or geometry is None:
            warnings.append(f"Skipped malformed or unsupported {source} primitive at index {index}.")
            continue
        feature_id = _feature_id(primitive, source, index)
        if feature_id in used_ids:
            feature_id = f"{feature_id}-{index}"
        used_ids.add(feature_id)
        score = _number(primitive.get("score")) if source == "detector" else None
        feature = {
            "feature_id": feature_id,
            "source": source,
            "source_id": primitive.get("id") if source == "label" else primitive.get("query_index"),
            "type": kind,
            "geometry": {key: _round(value) for key, value in geometry.items()},
            "confidence": _round(score) if score is not None else (1.0 if source == "label" else None),
            "evidence": "model prediction" if source == "detector" else "ParaCAD label (evaluation only)",
        }
        feature["metrics"] = _shape_metrics(feature, width, height)
        features.append(feature)
    return features, warnings


def _geometry_error(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Symmetric enough geometry distance to pair same-class primitives."""
    if first["type"] != second["type"]:
        return math.inf
    first_geometry, second_geometry = first["geometry"], second["geometry"]
    if first["type"] == "line":
        forward = [
            abs(first_geometry["x1"] - second_geometry["x1"]),
            abs(first_geometry["y1"] - second_geometry["y1"]),
            abs(first_geometry["x2"] - second_geometry["x2"]),
            abs(first_geometry["y2"] - second_geometry["y2"]),
        ]
        reverse = [
            abs(first_geometry["x1"] - second_geometry["x2"]),
            abs(first_geometry["y1"] - second_geometry["y2"]),
            abs(first_geometry["x2"] - second_geometry["x1"]),
            abs(first_geometry["y2"] - second_geometry["y1"]),
        ]
        return min(sum(forward), sum(reverse)) / 4
    values = [
        abs(first_geometry["cx"] - second_geometry["cx"]),
        abs(first_geometry["cy"] - second_geometry["cy"]),
        abs(first_geometry["radius"] - second_geometry["radius"]),
    ]
    if first["type"] == "arc":
        for key in ("start_param", "end_param"):
            values.append(abs((first_geometry[key] - second_geometry[key] + 180) % 360 - 180) / 180)
    return sum(values) / len(values)


def match_features(
    predictions: list[dict[str, Any]], labels: list[dict[str, Any]], threshold: float = 0.03
) -> list[dict[str, Any]]:
    """Greedily pair model features to labels for transparent evaluation context."""
    unmatched = set(range(len(labels)))
    matches: list[dict[str, Any]] = []
    for prediction in predictions:
        candidates = [(_geometry_error(prediction, labels[index]), index) for index in unmatched if labels[index]["type"] == prediction["type"]]
        if not candidates:
            continue
        error, label_index = min(candidates)
        if error <= threshold:
            unmatched.remove(label_index)
            matches.append(
                {
                    "predicted_feature_id": prediction["feature_id"],
                    "ground_truth_feature_id": labels[label_index]["feature_id"],
                    "geometry_mae_normalized": _round(error),
                }
            )
    return matches


def _cluster_points(points: list[tuple[str, tuple[float, float]]], tolerance: float) -> list[list[tuple[str, tuple[float, float]]]]:
    clusters: list[list[tuple[str, tuple[float, float]]]] = []
    for item in points:
        for cluster in clusters:
            if _distance(item[1], cluster[0][1]) <= tolerance:
                cluster.append(item)
                break
        else:
            clusters.append([item])
    return clusters


def analyze_topology(features: list[dict[str, Any]], endpoint_tolerance: float) -> dict[str, Any]:
    """Build a lightweight endpoint connectivity graph from detected geometry."""
    endpoint_rows: list[tuple[str, tuple[float, float]]] = []
    for feature in features:
        endpoint_rows.extend((feature["feature_id"], point) for point in _feature_points(feature))
    clusters = _cluster_points(endpoint_rows, endpoint_tolerance)
    adjacency: dict[str, set[str]] = {feature["feature_id"]: set() for feature in features}
    junctions: list[dict[str, Any]] = []
    for cluster in clusters:
        identifiers = sorted({identifier for identifier, _ in cluster})
        if len(identifiers) > 1:
            for first in identifiers:
                adjacency[first].update(identifier for identifier in identifiers if identifier != first)
        if len(cluster) > 1:
            x = sum(point[0] for _, point in cluster) / len(cluster)
            y = sum(point[1] for _, point in cluster) / len(cluster)
            junctions.append(
                {"feature_ids": identifiers, "location_normalized": {"x": _round(x), "y": _round(y)}, "endpoint_count": len(cluster)}
            )
    components: list[dict[str, Any]] = []
    unseen = set(adjacency)
    while unseen:
        start = unseen.pop()
        stack, component = [start], {start}
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in component:
                    component.add(neighbor)
                    unseen.discard(neighbor)
                    stack.append(neighbor)
        component_features = [feature for feature in features if feature["feature_id"] in component]
        degrees = {identifier: len(adjacency[identifier] & component) for identifier in component}
        only_lines = bool(component_features) and all(feature["type"] == "line" for feature in component_features)
        is_closed = only_lines and len(component) >= 3 and all(degree == 2 for degree in degrees.values())
        components.append(
            {
                "component_id": f"component-{len(components) + 1}",
                "feature_ids": sorted(component),
                "feature_count": len(component),
                "is_closed_line_profile": is_closed,
                "endpoint_degree": degrees,
            }
        )
    return {
        "endpoint_tolerance_normalized": endpoint_tolerance,
        "junctions": junctions,
        "components": components,
        "closed_line_profile_count": sum(component["is_closed_line_profile"] for component in components),
    }


def _compatible_under_mirror(first: dict[str, Any], second: dict[str, Any], axis: float, orientation: str, tolerance: float) -> bool:
    if first["type"] != second["type"]:
        return False
    first_center, second_center = feature_center(first), feature_center(second)
    if orientation == "vertical":
        expected = (2 * axis - first_center[0], first_center[1])
    else:
        expected = (first_center[0], 2 * axis - first_center[1])
    if _distance(expected, second_center) > tolerance:
        return False
    if first["type"] == "line":
        return abs(first["metrics"]["length_normalized"] - second["metrics"]["length_normalized"]) <= tolerance * 2
    return abs(first["geometry"]["radius"] - second["geometry"]["radius"]) <= tolerance


def find_symmetry(features: list[dict[str, Any]], tolerance: float = 0.015) -> list[dict[str, Any]]:
    """Propose only axes supported by at least two reflected feature pairs."""
    results: list[dict[str, Any]] = []
    for orientation, coordinate_index in (("vertical", 0), ("horizontal", 1)):
        candidates: Counter[float] = Counter()
        for first_index, first in enumerate(features):
            first_center = feature_center(first)
            for second in features[first_index + 1 :]:
                if first["type"] != second["type"]:
                    continue
                second_center = feature_center(second)
                if abs(first_center[1 - coordinate_index] - second_center[1 - coordinate_index]) <= tolerance:
                    candidates[round((first_center[coordinate_index] + second_center[coordinate_index]) / 2, 3)] += 1
        for axis, _ in candidates.most_common(4):
            pairs: list[list[str]] = []
            available = set(range(len(features)))
            for first_index, first in enumerate(features):
                if first_index not in available:
                    continue
                for second_index in sorted(available - {first_index}):
                    if _compatible_under_mirror(first, features[second_index], axis, orientation, tolerance):
                        pairs.append([first["feature_id"], features[second_index]["feature_id"]])
                        available.remove(first_index)
                        available.remove(second_index)
                        break
            if len(pairs) >= 2:
                results.append(
                    {
                        "axis": orientation,
                        "coordinate_normalized": axis,
                        "matched_pairs": pairs,
                        "pair_count": len(pairs),
                        "confidence": _round(min(0.95, 0.45 + 0.12 * len(pairs)), 3),
                        "interpretation": "geometric pattern proposal; verify against dimensions and views",
                    }
                )
    return results


def find_repeated_holes(features: list[dict[str, Any]], radius_tolerance: float = 0.004) -> list[dict[str, Any]]:
    circles = [feature for feature in features if feature["type"] == "circle"]
    groups: list[list[dict[str, Any]]] = []
    for circle in circles:
        for group in groups:
            if abs(circle["geometry"]["radius"] - group[0]["geometry"]["radius"]) <= radius_tolerance:
                group.append(circle)
                break
        else:
            groups.append([circle])
    patterns: list[dict[str, Any]] = []
    for group in groups:
        if len(group) < 2:
            continue
        centers = sorted((feature_center(feature), feature["feature_id"]) for feature in group)
        x_steps = [centers[index + 1][0][0] - centers[index][0][0] for index in range(len(centers) - 1)]
        patterns.append(
            {
                "pattern_id": f"hole-pattern-{len(patterns) + 1}",
                "feature_ids": [feature["feature_id"] for feature in group],
                "count": len(group),
                "radius_normalized": _round(sum(feature["geometry"]["radius"] for feature in group) / len(group)),
                "center_spacing_x_normalized": [_round(step) for step in x_steps],
                "interpretation": "same-radius circular-feature group; it may represent a repeated hole pattern",
            }
        )
    return patterns


def _parse_ocr(image_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    """Collect OCR candidates without making OCR a mandatory dependency."""
    warnings: list[str] = []
    try:
        import pytesseract  # type: ignore[import-not-found]
    except ImportError:
        return [], [], {}, ["OCR skipped: install pytesseract and the Tesseract executable to read drawing annotations."]
    try:
        data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
    except Exception as exc:  # Tesseract missing / inaccessible / malformed input
        return [], [], {}, [f"OCR skipped: {type(exc).__name__}: {exc}"]
    image = Image.open(image_path)
    width, height = image.size
    text_rows: list[dict[str, Any]] = []
    dimension_candidates: list[dict[str, Any]] = []
    title_rows: list[dict[str, Any]] = []
    measurement_pattern = re.compile(r"(?i)(?:[⌀Ø]|DIA\.?\s*)?(R)?\s*(\d+(?:\.\d+)?)")
    for index, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        confidence = _number(data.get("conf", [None])[index])
        if not text or confidence is None or confidence < 20:
            continue
        left, top = float(data["left"][index]), float(data["top"][index])
        row = {
            "annotation_id": f"ocr-{index}",
            "text": text,
            "confidence": _round(confidence / 100.0, 3),
            "bbox_pixels": {"x": _round(left, 1), "y": _round(top, 1), "width": _round(float(data["width"][index]), 1), "height": _round(float(data["height"][index]), 1)},
            "source": "ocr",
            "verification": "unverified text recognition",
        }
        text_rows.append(row)
        match = measurement_pattern.fullmatch(text.replace(" ", ""))
        if match:
            is_radius = bool(match.group(1)) or text.upper().startswith("R")
            is_diameter = "Ø" in text or "⌀" in text or text.upper().startswith("DIA")
            dimension_candidates.append(
                {
                    "dimension_id": f"ocr-dim-{index}",
                    "type": "radius" if is_radius else ("diameter" if is_diameter else "numeric_annotation"),
                    "value_text": text,
                    "value": _number(match.group(2)),
                    "unit": "unknown",
                    "source": "ocr",
                    "confidence": row["confidence"],
                    "verification": "unverified; no geometric association was inferred",
                }
            )
        if left / width >= 0.60 and top / height >= 0.60:
            title_rows.append(row)
    title_block = {"candidate_text": title_rows, "verification": "OCR candidates in lower-right image region; not a parsed title block"} if title_rows else {}
    return text_rows, dimension_candidates, title_block, warnings


def _label_annotations(record: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not record:
        return [], []
    constraints = []
    for index, constraint in enumerate(record.get("constraints", [])):
        source, target = constraint.get("source"), constraint.get("target")
        constraints.append(
            {
                "constraint_id": f"GT-constraint-{index}",
                "type": constraint.get("type", "unknown"),
                "feature_ids": [f"GT-{value}" for value in (source, target) if value],
                "point_types": {"source": constraint.get("pointType1"), "target": constraint.get("pointType2")},
                "source": "label",
                "verification": "ParaCAD ground truth; available only for evaluation",
            }
        )
    dimensions = []
    for index, dimension in enumerate(record.get("dimensions", [])):
        associated = [str(value) for value in dimension.get("associated_primitives", [])]
        dimensions.append(
            {
                "dimension_id": str(dimension.get("id", f"GT-dim-{index}")),
                "type": dimension.get("type", "unknown"),
                "value": _number(dimension.get("value")),
                "unit": dimension.get("unit", "unknown"),
                "references": dimension.get("refs", []),
                "feature_ids": [f"GT-{value}" for value in associated],
                "computed_geometry_value_normalized": _number(dimension.get("computed_from_geometry")),
                "source": "label",
                "verification": "ParaCAD ground truth; available only for evaluation",
            }
        )
    return constraints, dimensions


def run_checks(
    features: list[dict[str, Any]],
    topology: dict[str, Any],
    confidence_threshold: float,
    duplicate_tolerance: float,
    dimensions: list[dict[str, Any]] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    label_features: list[dict[str, Any]] | None = None,
    min_hole_edge_spacing: float | None = None,
) -> list[dict[str, Any]]:
    """Return transparent engineering-review flags, not autonomous design verdicts."""
    findings: list[dict[str, Any]] = []
    for feature in features:
        confidence = feature.get("confidence")
        if confidence is not None and confidence < confidence_threshold:
            findings.append(
                {
                    "rule_id": "low_detector_confidence",
                    "severity": "review",
                    "feature_ids": [feature["feature_id"]],
                    "message": f"Detector confidence {confidence:.2f} is below the review threshold {confidence_threshold:.2f}.",
                    "evidence": "model prediction score",
                }
            )
        if feature["type"] == "line" and feature["metrics"]["length_normalized"] < 0.003:
            findings.append(
                {
                    "rule_id": "very_short_line",
                    "severity": "review",
                    "feature_ids": [feature["feature_id"]],
                    "message": "Very short detected line; it may be a real detail, extension line, or duplicate/noise.",
                    "evidence": {"length_normalized": feature["metrics"]["length_normalized"]},
                }
            )
    for first_index, first in enumerate(features):
        for second in features[first_index + 1 :]:
            if _geometry_error(first, second) <= duplicate_tolerance:
                findings.append(
                    {
                        "rule_id": "near_duplicate_detection",
                        "severity": "review",
                        "feature_ids": [first["feature_id"], second["feature_id"]],
                        "message": "Two same-type detections are nearly coincident; retain one only after visual review.",
                        "evidence": {"geometry_mae_normalized": _round(_geometry_error(first, second))},
                    }
                )
    circles = [feature for feature in features if feature["type"] == "circle"]
    for first_index, first in enumerate(circles):
        for second in circles[first_index + 1 :]:
            separation = _distance(feature_center(first), feature_center(second))
            minimum = first["geometry"]["radius"] + second["geometry"]["radius"]
            if separation < minimum:
                findings.append(
                    {
                        "rule_id": "overlapping_circular_features",
                        "severity": "warning",
                        "feature_ids": [first["feature_id"], second["feature_id"]],
                        "message": "Circular-feature envelopes overlap in image space; check whether these are concentric features or a detector error.",
                        "evidence": {"center_distance_normalized": _round(separation), "sum_of_radii_normalized": _round(minimum)},
                    }
                )
            edge_spacing = separation - minimum
            if min_hole_edge_spacing is not None and edge_spacing < min_hole_edge_spacing:
                findings.append(
                    {
                        "rule_id": "hole_edge_spacing_below_review_threshold",
                        "severity": "warning",
                        "feature_ids": [first["feature_id"], second["feature_id"]],
                        "message": "Circular-feature edge spacing is below the configured image-space review threshold. Confirm callouts, scale, and manufacturing requirements.",
                        "evidence": {
                            "edge_spacing_normalized": _round(edge_spacing),
                            "configured_minimum_normalized": _round(min_hole_edge_spacing),
                        },
                    }
                )
    label_by_id = {feature["feature_id"]: feature for feature in (label_features or [])}
    for dimension in dimensions or []:
        feature_ids = [str(value) for value in dimension.get("feature_ids", [])]
        unresolved = [feature_id for feature_id in feature_ids if feature_id not in label_by_id]
        if unresolved and dimension.get("source") == "label":
            findings.append(
                {
                    "rule_id": "unresolved_dimension_reference",
                    "severity": "warning",
                    "feature_ids": unresolved,
                    "message": "A ground-truth dimension references geometry not present in the supplied label feature set.",
                    "evidence": {"dimension_id": dimension.get("dimension_id"), "unresolved_feature_ids": unresolved},
                }
            )
        if dimension.get("unit") in {None, "", "unknown"}:
            findings.append(
                {
                    "rule_id": "dimension_unit_unverified",
                    "severity": "info",
                    "feature_ids": feature_ids,
                    "message": "A dimension or OCR numeric candidate has no verified unit. Do not use it as a manufacturing measurement.",
                    "evidence": {"dimension_id": dimension.get("dimension_id"), "source": dimension.get("source")},
                }
            )
    dimension_groups: dict[tuple[str, tuple[str, ...], str], set[float]] = defaultdict(set)
    for dimension in dimensions or []:
        value = _number(dimension.get("value"))
        if value is not None and dimension.get("source") == "label":
            key = (str(dimension.get("type")), tuple(sorted(str(item) for item in dimension.get("feature_ids", []))), str(dimension.get("unit")))
            dimension_groups[key].add(round(value, 8))
    for (dimension_type, feature_ids, unit), values in dimension_groups.items():
        if len(values) > 1:
            findings.append(
                {
                    "rule_id": "conflicting_dimension_values",
                    "severity": "warning",
                    "feature_ids": list(feature_ids),
                    "message": "Multiple label dimensions assign different values to the same type and associated geometry. Check the drawing revision and source labels.",
                    "evidence": {"dimension_type": dimension_type, "unit": unit, "values": sorted(values)},
                }
            )
    for constraint in constraints or []:
        feature_ids = list(constraint.get("feature_ids", []))
        if not feature_ids or constraint.get("type") not in {"Horizontal", "Vertical"}:
            continue
        feature = label_by_id.get(feature_ids[0])
        if not feature or feature.get("type") != "line":
            continue
        orientation = feature.get("metrics", {}).get("orientation")
        expected = "horizontal" if constraint["type"] == "Horizontal" else "vertical"
        if orientation != expected:
            findings.append(
                {
                    "rule_id": "constraint_geometry_mismatch",
                    "severity": "warning",
                    "feature_ids": [feature_ids[0]],
                    "message": f"Ground-truth {constraint['type']} constraint does not agree with the stored line geometry orientation.",
                    "evidence": {"constraint_id": constraint.get("constraint_id"), "computed_orientation": orientation},
                }
            )
    if features and not topology["closed_line_profile_count"]:
        findings.append(
            {
                "rule_id": "no_closed_line_profile_detected",
                "severity": "info",
                "feature_ids": [],
                "message": "No closed profile was inferred from detected line endpoints. This can be normal for dimensioned views or incomplete detection.",
                "evidence": {"component_count": len(topology["components"])},
            }
        )
    return findings


def build_drawing_facts(
    image_path: Path,
    prediction_path: Path,
    ground_truth_jsonl: Path | None = None,
    use_ocr: bool = True,
    endpoint_tolerance: float = 0.012,
    duplicate_tolerance: float = 0.008,
    confidence_threshold: float = 0.55,
    min_hole_edge_spacing: float | None = None,
) -> dict[str, Any]:
    """Create a grounded facts document for engineer-facing Q&A and review."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction JSON not found: {prediction_path}")
    image = Image.open(image_path)
    width, height = image.size
    prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    coordinate_space = prediction_payload.get("coordinate_space")
    if coordinate_space != "normalized":
        raise ValueError("Only normalized detector output is currently supported; rerun paracad_primitive_detr.py predict.")
    predicted_features, warnings = make_features(prediction_payload.get("primitives", []), "detector", width, height)

    label_record: dict[str, Any] | None = None
    label_features: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []
    label_dimensions: list[dict[str, Any]] = []
    if ground_truth_jsonl is not None:
        if not ground_truth_jsonl.is_file():
            raise FileNotFoundError(f"Ground-truth JSONL not found: {ground_truth_jsonl}")
        label_record = _find_record(ground_truth_jsonl, image_path)
        label_features, label_warnings = make_features(label_record.get("primitives", []), "label", width, height)
        warnings.extend(label_warnings)
        constraints, label_dimensions = _label_annotations(label_record)

    ocr_text: list[dict[str, Any]] = []
    ocr_dimensions: list[dict[str, Any]] = []
    title_block: dict[str, Any] = {}
    if use_ocr:
        ocr_text, ocr_dimensions, title_block, ocr_warnings = _parse_ocr(image_path)
        warnings.extend(ocr_warnings)

    topology = analyze_topology(predicted_features, endpoint_tolerance)
    symmetries = find_symmetry(predicted_features)
    repeated_holes = find_repeated_holes(predicted_features)
    matches = match_features(predicted_features, label_features) if label_features else []
    findings = run_checks(
        predicted_features,
        topology,
        confidence_threshold,
        duplicate_tolerance,
        dimensions=label_dimensions + ocr_dimensions,
        constraints=constraints,
        label_features=label_features,
        min_hole_edge_spacing=min_hole_edge_spacing,
    )
    counts = Counter(feature["type"] for feature in predicted_features)
    confidence_values = [feature["confidence"] for feature in predicted_features if feature["confidence"] is not None]
    facts: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "drawing": {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "image_size_pixels": {"width": width, "height": height},
            "coordinate_space": "normalized_image_coordinates",
            "physical_scale": "unknown",
        },
        "provenance": {
            "prediction_path": str(prediction_path),
            "detector_model": prediction_payload.get("model", "unknown"),
            "checkpoint_epoch": prediction_payload.get("checkpoint_epoch"),
            "ground_truth_jsonl": str(ground_truth_jsonl) if ground_truth_jsonl else None,
            "analysis_warnings": warnings,
            "confidence_policy": {
                "detector": "score emitted by the image-to-primitive model; it is not a manufacturing confidence.",
                "label": "ParaCAD labels are retained separately and only when supplied for evaluation.",
                "ocr": "OCR output is unverified until an engineer confirms it against the drawing.",
            },
        },
        "summary": {
            "predicted_primitive_count": len(predicted_features),
            "predicted_type_counts": dict(sorted(counts.items())),
            "mean_detector_confidence": _round(sum(confidence_values) / len(confidence_values)) if confidence_values else None,
            "ground_truth_primitive_count": len(label_features) if label_record else None,
            "matched_prediction_count": len(matches) if label_record else None,
            "closed_line_profile_count": topology["closed_line_profile_count"],
            "rule_finding_count": len(findings),
        },
        "features": {
            "predicted_primitives": predicted_features,
            "ground_truth_primitives": label_features,
            "hole_candidates": [feature["feature_id"] for feature in predicted_features if feature["type"] == "circle"],
            "repeated_hole_patterns": repeated_holes,
            "symmetry_proposals": symmetries,
        },
        "topology": topology,
        "annotations": {
            "dimensions": label_dimensions + ocr_dimensions,
            "constraints": constraints,
            "ocr_text": ocr_text,
            "title_block": title_block,
        },
        "evaluation": {
            "matches": matches,
            "note": "This section exists only when ParaCAD ground truth was supplied. It must not be used as evidence for a new, unlabeled drawing.",
        },
        "checks": {
            "findings": findings,
            "policy": {
                "min_hole_edge_spacing_normalized": min_hole_edge_spacing,
                "limitations": "Rules evaluate image geometry and available labels only. GD&T, material, fit, scale, and manufacturing conformance require verified specifications.",
            },
        },
    }
    return facts


def write_facts(facts: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
