#!/usr/bin/env python3
"""Extract geometric, topology, and dimension features from ParaCAD Zarr data.

This reads ParaCAD's *structured CAD labels* from the Zarr archive.  It does
not infer geometry from image pixels: use its output as ground truth for
exploration, rule-based analysis, or training an image-to-CAD model.

Each JSONL output row describes one engineering drawing with:
  * primitive counts and a geometry bounding box;
  * line orientation/length and circle/arc radius statistics;
  * constraint-graph and dimension statistics; and
  * image/split/style metadata.

Examples (from the repository root):

  # Inspect 100 training drawings (the safe default).
  python extract_drawing_features.py --split train

  # Extract all validation drawings, including the individual primitives.
  python extract_drawing_features.py --split val --limit 0 --include-primitives \
      --output val_features.jsonl

  # Work with source-coordinate geometry instead of image-normalized geometry.
  python extract_drawing_features.py --coordinate-space native --limit 1000

Install dependencies once if needed:
  python -m pip install "zarr>=3" numpy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from cli_progress import RichProgress, add_progress_argument


PRIMITIVE_TYPES = {0: "unknown", 1: "line", 2: "circle", 3: "arc", 4: "point"}
CONSTRAINT_TYPES = {
    0: "unknown",
    1: "Coincident",
    2: "PointOnObject",
    3: "Horizontal",
    4: "Vertical",
    5: "Parallel",
    6: "Perpendicular",
    7: "Tangent",
    8: "Equal",
}
DIMENSION_TYPES = {0: "unknown", 1: "linear", 2: "diameter", 3: "radius", 4: "angular"}
SPLITS = {0: "train", 1: "val", 2: "test"}
STYLES = {0: "unknown", 1: "color1", 2: "color2", 3: "white"}


def finite_values(values: Iterable[float]) -> list[float]:
    """Return finite values as ordinary Python floats for strict JSON output."""
    return [float(value) for value in values if math.isfinite(float(value))]


def numeric_summary(values: Iterable[float]) -> Optional[dict[str, float]]:
    """Return compact statistics, or None when a quantity is not available."""
    valid = finite_values(values)
    if not valid:
        return None
    data = np.asarray(valid, dtype=np.float64)
    return {
        "count": int(data.size),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
    }


def named_counts(type_ids: np.ndarray, names: dict[int, str]) -> dict[str, int]:
    counts = Counter(int(value) for value in type_ids)
    return {name: int(counts.get(type_id, 0)) for type_id, name in names.items()}


def angle_to_radians(value: float) -> float:
    """ParaCAD stores its arc start and end parameters in degrees."""
    return math.radians(value)


def arc_bounds(cx: float, cy: float, radius: float, start: float, end: float) -> list[tuple[float, float]]:
    """Return arc endpoints plus any cardinal extrema contained by its CCW sweep."""
    if not all(math.isfinite(value) for value in (cx, cy, radius, start, end)) or radius < 0:
        return []

    start_rad = angle_to_radians(start) % (2 * math.pi)
    end_rad = angle_to_radians(end) % (2 * math.pi)
    sweep = (end_rad - start_rad) % (2 * math.pi)
    # Equal start/end values conventionally represent a full circle in CAD data.
    if math.isclose(sweep, 0.0, abs_tol=1e-8):
        sweep = 2 * math.pi

    angles = [start_rad, start_rad + sweep]
    for cardinal in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        distance = (cardinal - start_rad) % (2 * math.pi)
        if distance <= sweep + 1e-8:
            angles.append(start_rad + distance)
    return [(cx + radius * math.cos(angle), cy + radius * math.sin(angle)) for angle in angles]


def geometry_points(type_ids: np.ndarray, geometry: np.ndarray) -> list[tuple[float, float]]:
    """Return extent-defining points for lines, circles, and arcs."""
    points: list[tuple[float, float]] = []
    for primitive_type, row in zip(type_ids, geometry):
        kind = int(primitive_type)
        x1, y1, x2, y2, radius, start, end = (float(value) for value in row)
        if kind == 1:  # line
            points.extend(((x1, y1), (x2, y2)))
        elif kind == 2 and all(math.isfinite(value) for value in (x1, y1, radius)):  # circle
            points.extend(((x1 - radius, y1 - radius), (x1 + radius, y1 + radius)))
        elif kind == 3:  # arc
            points.extend(arc_bounds(x1, y1, radius, start, end))
    return [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]


def bounding_box(type_ids: np.ndarray, geometry: np.ndarray) -> Optional[dict[str, float]]:
    points = geometry_points(type_ids, geometry)
    if not points:
        return None
    xs, ys = zip(*points)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        "x_min": float(x_min),
        "y_min": float(y_min),
        "x_max": float(x_max),
        "y_max": float(y_max),
        "width": float(x_max - x_min),
        "height": float(y_max - y_min),
    }


def primitive_details(type_ids: np.ndarray, geometry: np.ndarray) -> list[dict[str, Any]]:
    """Convert numeric Zarr geometry into readable per-primitive CAD features."""
    result: list[dict[str, Any]] = []
    for index, (primitive_type, row) in enumerate(zip(type_ids, geometry)):
        kind = int(primitive_type)
        x1, y1, x2, y2, radius, start, end = (float(value) for value in row)
        feature: dict[str, Any] = {"local_index": index, "type": PRIMITIVE_TYPES.get(kind, "unknown")}
        def number_or_none(value: float) -> Optional[float]:
            return value if math.isfinite(value) else None

        if kind == 1:
            feature["geometry"] = {
                "x1": number_or_none(x1),
                "y1": number_or_none(y1),
                "x2": number_or_none(x2),
                "y2": number_or_none(y2),
            }
        elif kind == 2:
            feature["geometry"] = {"cx": number_or_none(x1), "cy": number_or_none(y1), "radius": number_or_none(radius)}
        elif kind == 3:
            feature["geometry"] = {
                "cx": number_or_none(x1),
                "cy": number_or_none(y1),
                "radius": number_or_none(radius),
                "start_param": number_or_none(start),
                "end_param": number_or_none(end),
            }
        else:
            feature["geometry"] = None
        result.append(feature)
    return result


def record_features(
    record_index: int,
    image_width: int,
    image_height: int,
    split_id: int,
    style_id: int,
    primitive_types: np.ndarray,
    primitive_geometry: np.ndarray,
    constraint_types: np.ndarray,
    constraint_source: np.ndarray,
    constraint_target: np.ndarray,
    dimension_types: np.ndarray,
    dimension_values: np.ndarray,
    dimension_from_geometry: np.ndarray,
    dimension_ref_counts: np.ndarray,
    coordinate_space: str,
    include_primitives: bool,
) -> dict[str, Any]:
    """Build one serializable feature record from contiguous Zarr table slices."""
    line_rows = primitive_geometry[primitive_types == 1]
    if line_rows.size:
        dx = line_rows[:, 2] - line_rows[:, 0]
        dy = line_rows[:, 3] - line_rows[:, 1]
        lengths = np.hypot(dx, dy)
        # Coordinates are normalized by max(image width, image height); this
        # threshold remains intentionally small in either supported space.
        tolerance = 1e-4 if coordinate_space == "normalized" else 1e-3
        orientations = {
            "horizontal": int(np.count_nonzero(np.abs(dy) <= tolerance)),
            "vertical": int(np.count_nonzero(np.abs(dx) <= tolerance)),
            "diagonal": int(np.count_nonzero((np.abs(dx) > tolerance) & (np.abs(dy) > tolerance))),
        }
    else:
        lengths = np.asarray([], dtype=np.float64)
        orientations = {"horizontal": 0, "vertical": 0, "diagonal": 0}

    circle_or_arc_rows = primitive_geometry[np.isin(primitive_types, (2, 3))]
    radii = circle_or_arc_rows[:, 4] if circle_or_arc_rows.size else np.asarray([], dtype=np.float64)

    usable_constraint_nodes = np.concatenate((constraint_source, constraint_target))
    usable_constraint_nodes = usable_constraint_nodes[usable_constraint_nodes >= 0]
    feature: dict[str, Any] = {
        "record_index": record_index,
        "image": {"width": image_width, "height": image_height, "style": STYLES.get(style_id, "unknown")},
        "split": SPLITS.get(split_id, "unknown"),
        "coordinate_space": coordinate_space,
        "primitive_counts": named_counts(primitive_types, PRIMITIVE_TYPES),
        "geometry_bbox": bounding_box(primitive_types, primitive_geometry),
        "line_features": {"length": numeric_summary(lengths), "orientation_counts": orientations},
        "curve_features": {
            "radius": numeric_summary(radii),
            "circle_count": int(np.count_nonzero(primitive_types == 2)),
            "arc_count": int(np.count_nonzero(primitive_types == 3)),
        },
        "constraint_features": {
            "count": int(constraint_types.size),
            "type_counts": named_counts(constraint_types, CONSTRAINT_TYPES),
            "referenced_primitive_count": int(np.unique(usable_constraint_nodes).size),
        },
        "dimension_features": {
            "count": int(dimension_types.size),
            "type_counts": named_counts(dimension_types, DIMENSION_TYPES),
            "label_value": numeric_summary(dimension_values),
            "computed_geometry_value": numeric_summary(dimension_from_geometry),
            "reference_count": numeric_summary(dimension_ref_counts),
        },
    }
    if include_primitives:
        feature["primitives"] = primitive_details(primitive_types, primitive_geometry)
    return feature


def open_dataset(path: Path) -> Any:
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError('Missing dependency "zarr". Install it with: python -m pip install "zarr>=3" numpy') from exc
    return zarr.open_group(str(path), mode="r")


def required_array(root: Any, path: str) -> Any:
    try:
        return root[path]
    except KeyError as exc:
        raise RuntimeError(f"The dataset does not contain the required array: /{path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract one compact JSON feature record per ParaCAD drawing.")
    parser.add_argument("--dataset", type=Path, default=Path("ParaCAD_full_v3.zarr"), help="Input Zarr directory.")
    parser.add_argument("--output", type=Path, default=Path("drawing_features.jsonl"), help="JSONL output path.")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="train", help="Dataset split to analyze.")
    parser.add_argument("--start", type=int, default=0, help="First record index to consider (zero based).")
    parser.add_argument("--limit", type=int, default=100, help="Number of matching records (0 means no limit).")
    parser.add_argument("--batch-size", type=int, default=512, help="Records loaded at a time.")
    parser.add_argument(
        "--coordinate-space",
        choices=("normalized", "native"),
        default="normalized",
        help="Use image-normalized geometry or native ParaCAD label coordinates.",
    )
    parser.add_argument("--include-primitives", action="store_true", help="Also include readable geometry for every primitive.")
    add_progress_argument(parser)
    args = parser.parse_args()

    if args.start < 0 or args.limit < 0 or args.batch_size <= 0:
        parser.error("--start and --limit must be non-negative, and --batch-size must be positive.")
    if not args.dataset.is_dir():
        parser.error(f"Zarr dataset directory not found: {args.dataset}")

    try:
        root = open_dataset(args.dataset)
        records = {
            "width": required_array(root, "records/image_width"),
            "height": required_array(root, "records/image_height"),
            "split": required_array(root, "records/split"),
            "style": required_array(root, "records/style"),
            "primitive_offset": required_array(root, "records/primitive_offset"),
            "primitive_count": required_array(root, "records/primitive_count"),
            "constraint_offset": required_array(root, "records/constraint_offset"),
            "constraint_count": required_array(root, "records/constraint_count"),
            "dimension_offset": required_array(root, "records/dimension_offset"),
            "dimension_count": required_array(root, "records/dimension_count"),
        }
        primitive_geometry_name = "primitives/geometry_norm" if args.coordinate_space == "normalized" else "primitives/geometry"
        tables = {
            "primitive_type": required_array(root, "primitives/type"),
            "primitive_geometry": required_array(root, primitive_geometry_name),
            "constraint_type": required_array(root, "constraints/type"),
            "constraint_source": required_array(root, "constraints/source_local_index"),
            "constraint_target": required_array(root, "constraints/target_local_index"),
            "dimension_type": required_array(root, "dimensions/type"),
            "dimension_value": required_array(root, "dimensions/value"),
            "dimension_computed": required_array(root, "dimensions/computed_from_geometry"),
            "dimension_ref_count": required_array(root, "dimensions/ref_count"),
        }
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    record_count = int(records["split"].shape[0])
    if args.start >= record_count:
        parser.error(f"--start ({args.start}) is outside the dataset, which has {record_count:,} records.")

    wanted_split = None if args.split == "all" else next(split_id for split_id, name in SPLITS.items() if name == args.split)
    written = 0
    scanned = args.start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress = RichProgress(enabled=not args.no_progress).start()
    task_total = args.limit if args.limit else record_count - args.start
    task_label = f"Extracting {args.split} feature rows" if args.limit else f"Scanning {args.split} drawing features"
    task = progress.add_task(task_label, total=task_total)

    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        while scanned < record_count and (args.limit == 0 or written < args.limit):
            stop = min(scanned + args.batch_size, record_count)
            split_ids = np.asarray(records["split"][scanned:stop])
            selected = np.flatnonzero(split_ids == wanted_split) if wanted_split is not None else np.arange(stop - scanned)
            if args.limit:
                selected = selected[: args.limit - written]
            if selected.size:
                record_arrays = {name: np.asarray(array[scanned:stop]) for name, array in records.items()}
                # Rows for each child table are contiguous, so fetch each
                # selected batch in three reads instead of one read per drawing.
                primitive_start = int(record_arrays["primitive_offset"][selected].min())
                primitive_end = int(
                    (record_arrays["primitive_offset"][selected] + record_arrays["primitive_count"][selected]).max()
                )
                constraint_start = int(record_arrays["constraint_offset"][selected].min())
                constraint_end = int(
                    (record_arrays["constraint_offset"][selected] + record_arrays["constraint_count"][selected]).max()
                )
                dimension_start = int(record_arrays["dimension_offset"][selected].min())
                dimension_end = int(
                    (record_arrays["dimension_offset"][selected] + record_arrays["dimension_count"][selected]).max()
                )
                primitive_type_block = np.asarray(tables["primitive_type"][primitive_start:primitive_end])
                primitive_geometry_block = np.asarray(tables["primitive_geometry"][primitive_start:primitive_end])
                constraint_type_block = np.asarray(tables["constraint_type"][constraint_start:constraint_end])
                constraint_source_block = np.asarray(tables["constraint_source"][constraint_start:constraint_end])
                constraint_target_block = np.asarray(tables["constraint_target"][constraint_start:constraint_end])
                dimension_type_block = np.asarray(tables["dimension_type"][dimension_start:dimension_end])
                dimension_value_block = np.asarray(tables["dimension_value"][dimension_start:dimension_end])
                dimension_computed_block = np.asarray(tables["dimension_computed"][dimension_start:dimension_end])
                dimension_ref_count_block = np.asarray(tables["dimension_ref_count"][dimension_start:dimension_end])

                for local_index_value in selected:
                    local_index = int(local_index_value)
                    primitive_offset = int(record_arrays["primitive_offset"][local_index])
                    primitive_count = int(record_arrays["primitive_count"][local_index])
                    constraint_offset = int(record_arrays["constraint_offset"][local_index])
                    constraint_count = int(record_arrays["constraint_count"][local_index])
                    dimension_offset = int(record_arrays["dimension_offset"][local_index])
                    dimension_count = int(record_arrays["dimension_count"][local_index])

                    primitive_slice = slice(primitive_offset - primitive_start, primitive_offset - primitive_start + primitive_count)
                    constraint_slice = slice(constraint_offset - constraint_start, constraint_offset - constraint_start + constraint_count)
                    dimension_slice = slice(dimension_offset - dimension_start, dimension_offset - dimension_start + dimension_count)
                    feature = record_features(
                        record_index=scanned + local_index,
                        image_width=int(record_arrays["width"][local_index]),
                        image_height=int(record_arrays["height"][local_index]),
                        split_id=int(record_arrays["split"][local_index]),
                        style_id=int(record_arrays["style"][local_index]),
                        primitive_types=primitive_type_block[primitive_slice],
                        primitive_geometry=primitive_geometry_block[primitive_slice],
                        constraint_types=constraint_type_block[constraint_slice],
                        constraint_source=constraint_source_block[constraint_slice],
                        constraint_target=constraint_target_block[constraint_slice],
                        dimension_types=dimension_type_block[dimension_slice],
                        dimension_values=dimension_value_block[dimension_slice],
                        dimension_from_geometry=dimension_computed_block[dimension_slice],
                        dimension_ref_counts=dimension_ref_count_block[dimension_slice],
                        coordinate_space=args.coordinate_space,
                        include_primitives=args.include_primitives,
                    )
                    output.write(json.dumps(feature, allow_nan=False, separators=(",", ":")) + "\n")
                    written += 1
            scanned = stop
            if task is not None:
                completed = written if args.limit else scanned - args.start
                progress.update(task, completed=completed, description=f"Extracting: {written:,} feature rows written")
            elif scanned % (args.batch_size * 10) == 0 or scanned == record_count:
                print(f"[PROGRESS] scanned {scanned:,}/{record_count:,}; wrote {written:,}", file=sys.stderr)

    progress.complete(task, f"Feature extraction complete: {written:,} rows")
    progress.stop()
    print(f"[DONE] Wrote {written:,} feature records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
