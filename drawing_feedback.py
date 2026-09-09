#!/usr/bin/env python3
"""Capture engineer corrections and export high-confidence local labels.

Feedback is append-only JSONL so that a reviewer decision is auditable. It is
never silently merged into source ParaCAD labels or a training set.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cli_progress import RichProgress, add_progress_argument


VALID_STATUSES = {"accepted", "rejected", "corrected", "missing"}
GEOMETRY_KEYS = {
    "line": {"x1", "y1", "x2", "y2"},
    "circle": {"cx", "cy", "radius"},
    "arc": {"cx", "cy", "radius", "start_param", "end_param"},
}


def load_facts(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Facts file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_geometry(raw: str, primitive_type: str) -> dict[str, float]:
    try:
        geometry = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --geometry JSON: {exc}") from exc
    if primitive_type not in GEOMETRY_KEYS or not isinstance(geometry, dict):
        raise ValueError("A supported --primitive-type and JSON object geometry are required.")
    missing = GEOMETRY_KEYS[primitive_type] - set(geometry)
    if missing:
        raise ValueError(f"Geometry for {primitive_type} is missing: {', '.join(sorted(missing))}")
    normalized: dict[str, float] = {}
    for key in GEOMETRY_KEYS[primitive_type]:
        try:
            normalized[key] = float(geometry[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Geometry value {key!r} must be numeric.") from exc
    return normalized


def append_feedback(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(event, ensure_ascii=False) + "\n")


def feedback_for_drawing(path: Path, drawing_path: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("drawing_image_path") == drawing_path:
                event["_line_number"] = line_number
                rows.append(event)
    return rows


def export_labels(facts: dict[str, Any], feedback: list[dict[str, Any]], include_unreviewed: bool) -> dict[str, Any]:
    predicted = {feature["feature_id"]: feature for feature in facts.get("features", {}).get("predicted_primitives", [])}
    latest: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for event in feedback:
        if event["status"] == "missing":
            missing.append(event)
        elif event.get("feature_id"):
            latest[event["feature_id"]] = event
    primitives: list[dict[str, Any]] = []
    for feature_id, feature in predicted.items():
        event = latest.get(feature_id)
        if event and event["status"] == "rejected":
            continue
        if not event and not include_unreviewed:
            continue
        geometry = event.get("corrected_geometry") if event and event["status"] == "corrected" else feature["geometry"]
        primitives.append(
            {
                "id": f"review-{feature_id}",
                "type": feature["type"],
                "geometry": geometry,
                "review_status": event["status"] if event else "unreviewed",
                "source_feature_id": feature_id,
            }
        )
    for index, event in enumerate(missing):
        primitives.append(
            {
                "id": f"review-missing-{index + 1}",
                "type": event["primitive_type"],
                "geometry": event["corrected_geometry"],
                "review_status": "missing_added",
                "source_feature_id": None,
            }
        )
    drawing = facts["drawing"]
    return {
        "schema_version": "engineer-feedback-labels/v1",
        "image": drawing["image_name"],
        "image_path": drawing["image_path"],
        "image_width": drawing["image_size_pixels"]["width"],
        "image_height": drawing["image_size_pixels"]["height"],
        "coordinate_space": "normalized_image_coordinates",
        "primitives": primitives,
        "feedback_event_count": len(feedback),
        "warning": "These are local engineer-reviewed labels. Convert them into the training format expected by a trainer; do not mix them with unreviewed detector predictions.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture auditable engineer feedback for drawing-model improvement.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add = subparsers.add_parser("add", help="Append one reviewer decision.")
    add.add_argument("--facts", type=Path, required=True)
    add.add_argument("--feedback", type=Path, default=Path("drawing_feedback.jsonl"))
    add.add_argument("--status", choices=sorted(VALID_STATUSES), required=True)
    add.add_argument("--feature-id", help="Required for accepted, rejected, or corrected detections.")
    add.add_argument("--primitive-type", choices=("line", "circle", "arc"), help="Required for missing geometry and corrected geometry validation.")
    add.add_argument("--geometry", help="Corrected/missing geometry JSON in normalized image coordinates.")
    add.add_argument("--note", default="")
    add.add_argument("--reviewer", default="unspecified")
    add_progress_argument(add)
    summary = subparsers.add_parser("summary", help="Summarize feedback for one drawing.")
    summary.add_argument("--facts", type=Path, required=True)
    summary.add_argument("--feedback", type=Path, default=Path("drawing_feedback.jsonl"))
    add_progress_argument(summary)
    export = subparsers.add_parser("export-labels", help="Export reviewed primitives as a portable custom-label JSONL row.")
    export.add_argument("--facts", type=Path, required=True)
    export.add_argument("--feedback", type=Path, default=Path("drawing_feedback.jsonl"))
    export.add_argument("--output", type=Path, default=Path("reviewed_labels.jsonl"))
    export.add_argument("--include-unreviewed", action="store_true", help="Include unreviewed predictions. Off by default to preserve label quality.")
    add_progress_argument(export)
    args = parser.parse_args()
    progress = RichProgress(enabled=not args.no_progress).start()
    task = progress.add_task("Loading drawing facts and reviewer feedback", total=2)
    try:
        facts = load_facts(args.facts)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        progress.stop()
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    progress.advance(task)
    drawing_path = str(facts.get("drawing", {}).get("image_path", ""))
    if not drawing_path:
        progress.stop()
        print("[ERROR] Facts document has no drawing.image_path.", file=sys.stderr)
        return 2
    existing = feedback_for_drawing(args.feedback, drawing_path)
    progress.complete(task, "Facts and feedback loaded")
    progress.stop()
    if args.command == "summary":
        counts: dict[str, int] = {status: 0 for status in sorted(VALID_STATUSES)}
        for event in existing:
            counts[event.get("status", "unknown")] = counts.get(event.get("status", "unknown"), 0) + 1
        print(json.dumps({"drawing_image_path": drawing_path, "feedback_file": str(args.feedback), "event_count": len(existing), "status_counts": counts}, indent=2))
        return 0
    if args.command == "export-labels":
        labels = export_labels(facts, existing, args.include_unreviewed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(labels, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[DONE] Wrote {args.output} with {len(labels['primitives'])} reviewed primitives.")
        return 0
    if args.status == "missing":
        if not args.primitive_type or not args.geometry:
            print("[ERROR] missing feedback requires --primitive-type and --geometry.", file=sys.stderr)
            return 2
        feature_id = None
        geometry = parse_geometry(args.geometry, args.primitive_type)
    else:
        if not args.feature_id:
            print(f"[ERROR] {args.status} feedback requires --feature-id.", file=sys.stderr)
            return 2
        feature = next((item for item in facts.get("features", {}).get("predicted_primitives", []) if item.get("feature_id") == args.feature_id), None)
        if feature is None:
            print(f"[ERROR] Unknown predicted feature ID: {args.feature_id}", file=sys.stderr)
            return 2
        feature_id = args.feature_id
        args.primitive_type = args.primitive_type or feature["type"]
        if args.status == "corrected":
            if not args.geometry:
                print("[ERROR] corrected feedback requires --geometry.", file=sys.stderr)
                return 2
            geometry = parse_geometry(args.geometry, args.primitive_type)
        else:
            geometry = None
    event = {
        "schema_version": "drawing-feedback/v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "drawing_image_path": drawing_path,
        "facts_path": str(args.facts),
        "status": args.status,
        "feature_id": feature_id,
        "primitive_type": args.primitive_type,
        "corrected_geometry": geometry,
        "note": args.note,
        "reviewer": args.reviewer,
    }
    append_feedback(args.feedback, event)
    print(f"[DONE] Appended {args.status} feedback to {args.feedback}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
