"""Small local retrieval index for approved engineering references.

The index intentionally uses inspectable lexical scoring rather than a remote
embedding service. This keeps proprietary standards and process notes local,
and makes every response cite an exact source file and chunk ID.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional


SCHEMA_VERSION = "engineering-knowledge/v1"
SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".json"}
EXCLUDED_DIRECTORY_NAMES = {".git", "__pycache__", ".pytest_cache", "node_modules"}
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9._/+\-]{1,}", re.IGNORECASE)
STOP_WORDS = {"and", "are", "for", "from", "into", "not", "the", "this", "that", "with", "when", "where", "what", "which", "will", "shall"}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if token.lower() not in STOP_WORDS]


def _chunks(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    rows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        rows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return rows


def build_index(
    source_dir: Path,
    chunk_chars: int = 1400,
    progress: Optional[Callable[[int, int, Path], None]] = None,
) -> tuple[dict[str, Any], list[str]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Knowledge source directory not found: {source_dir}")
    warnings: list[str] = []
    chunks: list[dict[str, Any]] = []
    source_files: list[Path] = []
    for root, directories, file_names in os.walk(source_dir):
        directories[:] = [
            directory
            for directory in directories
            if directory not in EXCLUDED_DIRECTORY_NAMES and not directory.lower().endswith(".zarr")
        ]
        source_files.extend(Path(root) / file_name for file_name in file_names)
    supported_files = [path for path in sorted(source_files) if path.suffix.lower() in SUPPORTED_SUFFIXES]
    for file_number, path in enumerate(supported_files, start=1):
        text: Optional[str] = None
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except OSError as exc:
                warnings.append(f"Skipped {path}: {exc}")
        except OSError as exc:
            warnings.append(f"Skipped {path}: {exc}")
        if text is not None:
            for offset, chunk in enumerate(_chunks(text, chunk_chars)):
                chunks.append(
                    {
                        "chunk_id": f"K{len(chunks) + 1}",
                        "source": str(path.relative_to(source_dir)).replace("\\", "/"),
                        "chunk_index": offset,
                        "text": chunk,
                        "token_counts": dict(Counter(_tokens(chunk))),
                    }
                )
        if progress is not None:
            progress(file_number, len(supported_files), path)
    index = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_directory": str(source_dir),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "limitations": "Lexical retrieval over approved local text files. PDF, DOCX, and image-only standards must be converted or added through a dedicated parser before indexing.",
    }
    return index, warnings


def write_index(index: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class KnowledgeIndex:
    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported knowledge index schema.")
        self.payload = payload
        self.chunks = [chunk for chunk in payload.get("chunks", []) if isinstance(chunk, dict)]
        self.document_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            self.document_frequency.update(set((chunk.get("token_counts") or {}).keys()))

    @classmethod
    def load(cls, index_path: Path) -> "KnowledgeIndex":
        if not index_path.is_file():
            raise FileNotFoundError(f"Knowledge index not found: {index_path}")
        return cls(json.loads(index_path.read_text(encoding="utf-8")))

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        terms = _tokens(query)
        if not terms:
            return {"error": "Provide a specific engineering question or terms to search the approved reference index."}
        results: list[tuple[float, dict[str, Any]]] = []
        count = max(1, len(self.chunks))
        for chunk in self.chunks:
            term_counts = chunk.get("token_counts") or {}
            total = max(1, sum(int(value) for value in term_counts.values()))
            score = 0.0
            for term in terms:
                frequency = int(term_counts.get(term, 0))
                if frequency:
                    inverse_frequency = math.log((count + 1) / (self.document_frequency.get(term, 0) + 1)) + 1
                    score += (frequency / total) * inverse_frequency
            if query.lower() in str(chunk.get("text", "")).lower():
                score += 0.5
            if score > 0:
                results.append((score, chunk))
        results.sort(key=lambda row: row[0], reverse=True)
        return {
            "query": query,
            "index_source_directory": self.payload.get("source_directory"),
            "result_count": len(results),
            "results": [
                {"citation": f"[{chunk['chunk_id']}]", "source": chunk.get("source"), "excerpt": chunk.get("text"), "score": round(score, 5)}
                for score, chunk in results[: max(1, min(10, int(limit)))]
            ],
            "warning": "Retrieved text is an approved reference candidate. Confirm revision, applicability, and governing requirements before approving a drawing.",
        }
