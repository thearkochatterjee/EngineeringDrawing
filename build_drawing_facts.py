#!/usr/bin/env python3
"""Create a grounded ``drawing_facts.json`` document from a model prediction.

Example:
  python build_drawing_facts.py --image drawing.png --prediction predicted_primitives.json \
      --output drawing_facts.json

Add ``--ground-truth-jsonl ParaCAD_processed_v2/train.jsonl`` only for
evaluation drawings.  Ground truth is deliberately marked separately from
model predictions, so it cannot be mistaken for information read from a new
drawing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cli_progress import RichProgress, add_progress_argument
from drawing_facts import build_drawing_facts, write_facts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build engineer-facing facts from a drawing image and primitive prediction JSON.")
    parser.add_argument("--image", type=Path, required=True, help="Original drawing image used for prediction.")
    parser.add_argument("--prediction", type=Path, required=True, help="JSON created by paracad_primitive_detr.py predict.")
    parser.add_argument("--output", type=Path, default=Path("drawing_facts.json"))
    parser.add_argument("--ground-truth-jsonl", type=Path, help="Optional ParaCAD JSONL. Use only to evaluate a known dataset drawing.")
    parser.add_argument("--no-ocr", action="store_true", help="Do not attempt optional Tesseract OCR.")
    parser.add_argument("--endpoint-tolerance", type=float, default=0.012, help="Normalized endpoint distance used for topology.")
    parser.add_argument("--duplicate-tolerance", type=float, default=0.008, help="Normalized geometry MAE used for duplicate flags.")
    parser.add_argument("--confidence-threshold", type=float, default=0.55, help="Detector scores below this value receive a review flag.")
    parser.add_argument("--min-hole-edge-spacing", type=float, help="Optional normalized image-space hole edge-spacing review threshold; not a physical manufacturing value.")
    add_progress_argument(parser)
    args = parser.parse_args()
    if args.endpoint_tolerance <= 0 or args.duplicate_tolerance <= 0 or not 0 <= args.confidence_threshold <= 1 or (args.min_hole_edge_spacing is not None and args.min_hole_edge_spacing < 0):
        parser.error("Tolerances must be positive and --confidence-threshold must be in [0, 1].")
    progress = RichProgress(enabled=not args.no_progress).start()
    task = progress.add_task("Building drawing facts", total=2)
    try:
        facts = build_drawing_facts(
            args.image,
            args.prediction,
            ground_truth_jsonl=args.ground_truth_jsonl,
            use_ocr=not args.no_ocr,
            endpoint_tolerance=args.endpoint_tolerance,
            duplicate_tolerance=args.duplicate_tolerance,
            confidence_threshold=args.confidence_threshold,
            min_hole_edge_spacing=args.min_hole_edge_spacing,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        progress.stop()
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    progress.advance(task)
    write_facts(facts, args.output)
    progress.complete(task, "Drawing facts complete")
    progress.stop()
    summary = facts["summary"]
    print(
        "[DONE] Wrote {} — {} predicted primitives, {} findings{}.".format(
            args.output,
            summary["predicted_primitive_count"],
            summary["rule_finding_count"],
            f", {summary['matched_prediction_count']} label matches" if summary["matched_prediction_count"] is not None else "",
        )
    )
    for warning in facts["provenance"]["analysis_warnings"]:
        print(f"[WARN] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
