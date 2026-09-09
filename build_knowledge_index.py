#!/usr/bin/env python3
"""Index approved local engineering-reference text for grounded copilot retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cli_progress import RichProgress, add_progress_argument
from engineering_knowledge import build_index, write_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an inspectable local index of approved engineering text references.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing approved .txt, .md, .csv, or .json reference files.")
    parser.add_argument("--output", type=Path, default=Path("engineering_knowledge.json"))
    parser.add_argument("--chunk-chars", type=int, default=1400)
    add_progress_argument(parser)
    args = parser.parse_args()
    if args.chunk_chars < 400:
        parser.error("--chunk-chars must be at least 400.")
    progress = RichProgress(enabled=not args.no_progress).start()
    task_id: int | None = None

    def report(completed: int, total: int, path: Path) -> None:
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task("Indexing approved reference files", total=total)
        progress.update(task_id, completed=completed, description=f"Indexing: {path.name}")

    try:
        index, warnings = build_index(args.source_dir, args.chunk_chars, progress=report)
    except FileNotFoundError as exc:
        progress.stop()
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    write_index(index, args.output)
    progress.complete(task_id, f"Knowledge index complete: {index['chunk_count']:,} chunks")
    progress.stop()
    print(f"[DONE] Wrote {args.output} with {index['chunk_count']} chunks.")
    for warning in warnings:
        print(f"[WARN] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
