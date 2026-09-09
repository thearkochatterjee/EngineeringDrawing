#!/usr/bin/env python3
"""Export reviewed drawing-facts examples for supervised copilot fine-tuning.

The output is standard JSONL with a ``messages`` array. It teaches a text model
to explain verified review outcomes and cite evidence; it does *not* train
primitive recognition from pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cli_progress import RichProgress, add_progress_argument
from drawing_feedback import feedback_for_drawing, load_facts


SYSTEM = (
    "You are an engineering-drawing review assistant. Use only supplied evidence, "
    "distinguish predictions from reviewer-confirmed labels, and never invent units, tolerances, or manufacturing approval."
)


def compact_facts(facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "drawing": facts.get("drawing", {}),
        "summary": facts.get("summary", {}),
        "review_findings": facts.get("checks", {}).get("findings", []),
        "predicted_primitives": [
            {"feature_id": feature.get("feature_id"), "type": feature.get("type"), "confidence": feature.get("confidence"), "geometry": feature.get("geometry")}
            for feature in facts.get("features", {}).get("predicted_primitives", [])
        ],
    }


def example_for(facts: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    user = "Review this drawing evidence and summarize the engineer-confirmed corrections. Cite feature IDs.\n\n"
    user += json.dumps({"facts": compact_facts(facts), "reviewer_feedback": events}, ensure_ascii=False)
    confirmed = [
        {
            "feature_id": event.get("feature_id"),
            "status": event.get("status"),
            "primitive_type": event.get("primitive_type"),
            "corrected_geometry": event.get("corrected_geometry"),
            "note": event.get("note"),
        }
        for event in events
    ]
    assistant = json.dumps(
        {
            "review_outcome": "Engineer-confirmed feedback only; unreviewed detector predictions remain unverified.",
            "confirmed_feedback": confirmed,
            "next_step": "Use reviewed examples for evaluation and future training only after checking image/label provenance.",
        },
        ensure_ascii=False,
    )
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create audited SFT examples from drawing facts and engineer feedback.")
    parser.add_argument("--facts", type=Path, action="append", required=True, help="Facts JSON. Repeat --facts for each reviewed drawing.")
    parser.add_argument("--feedback", type=Path, required=True, help="Append-only feedback JSONL created by drawing_feedback.py.")
    parser.add_argument("--output", type=Path, default=Path("copilot_sft.jsonl"))
    parser.add_argument("--include-empty-feedback", action="store_true", help="Include facts files with no reviewer feedback; normally excluded to protect training quality.")
    add_progress_argument(parser)
    args = parser.parse_args()
    examples: list[dict[str, Any]] = []
    progress = RichProgress(enabled=not args.no_progress).start()
    facts_task = progress.add_task("Preparing copilot SFT examples", total=len(args.facts))
    for facts_path in args.facts:
        try:
            facts = load_facts(facts_path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            progress.stop()
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        drawing_path = str(facts.get("drawing", {}).get("image_path", ""))
        events = feedback_for_drawing(args.feedback, drawing_path)
        if events or args.include_empty_feedback:
            examples.append(example_for(facts, events))
        progress.advance(facts_task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_task = progress.add_task("Writing copilot SFT JSONL", total=len(examples))
    with args.output.open("w", encoding="utf-8") as destination:
        for example in examples:
            destination.write(json.dumps(example, ensure_ascii=False) + "\n")
            progress.advance(write_task)
    progress.complete(facts_task, f"Prepared {len(examples):,} reviewed examples")
    progress.complete(write_task, "Copilot SFT JSONL complete")
    progress.stop()
    print(f"[DONE] Wrote {args.output} with {len(examples)} reviewed SFT examples.")
    if not examples:
        print("[WARN] No matching reviewer feedback was found. Capture accepted/rejected/corrected/missing decisions before fine-tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
