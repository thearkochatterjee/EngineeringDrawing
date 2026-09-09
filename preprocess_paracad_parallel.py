#!/usr/bin/env python3
"""
Fast ParaCAD preprocessing.

This script recursively scans a dataset folder containing JSON/JSONL files,
images, and optional zip files, then converts ParaCAD conversation-style labels
into structured JSONL suitable for training.

Speed choices:
  - Parallel zip extraction with a small thread pool.
  - Parallel JSON parsing with processes.
  - Worker-local output shards to avoid one shared write bottleneck.
  - Deterministic hash split, so records do not need to be held in RAM.
  - Fast image-size reads from file headers where possible.
  - Optional orjson / ijson use when installed.

Example:
  python preprocess_paracad_parallel.py ^
    --input "D:\\Datasets\\ParaCAD" ^
    --output "D:\\Datasets\\ParaCAD_processed" ^
    --extract-zips ^
    --normalize ^
    --workers 12 ^
    --zip-workers 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import struct
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from cli_progress import RichProgress, add_progress_argument

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None

try:
    import ijson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ijson = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None


LINE_RE = re.compile(r"^(Line\d+):\s*<Line>\s*([^<]+?)\s*</Line>\s*$", re.M)
CIRCLE_RE = re.compile(r"^(Circle\d+):\s*<Circle>\s*([^<]+?)\s*</Circle>\s*$", re.M)
ARC_RE = re.compile(r"^(Arc\d+):\s*<Arc>\s*([^<]+?)\s*</Arc>\s*$", re.M)

CONSTRAINT_RE = re.compile(
    r"^\((Coincident|PointOnObject|Horizontal|Vertical|Parallel|Perpendicular|Tangent|Equal),\s*"
    r"([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)\s*$",
    re.M,
)

DIMENSION_RE = re.compile(
    r"^\((Linear|Diameter|Radius|Angular),\s*\[(.*?)\],\s*([-+]?\d+(?:\.\d+)?)\)\s*$",
    re.M,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
JSON_EXTS = {".json", ".jsonl"}


G_IMAGE_INDEX: Dict[str, str] = {}
G_SEARCH_ROOTS: List[str] = []
G_SOURCE_ROOT: str = ""
G_OUTPUT_ROOT: str = ""
G_SHARD_ROOT: str = ""
G_NORMALIZE: bool = False
G_READ_IMAGE_SIZES: bool = False
G_TRAIN_RATIO: float = 0.90
G_VAL_RATIO: float = 0.05
G_WRITE_ALL: bool = True


def json_loads(data: str | bytes) -> Any:
    if orjson is not None:
        return orjson.loads(data)
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    return json.loads(data)


def json_dumps_line(obj: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(obj) + b"\n"
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def parse_scalar(x: Any) -> Any:
    if not isinstance(x, str):
        return x

    x = x.strip()
    if x == "True":
        return True
    if x == "False":
        return False
    if x == "None":
        return None

    try:
        if re.fullmatch(r"[-+]?\d+", x):
            return int(x)
        return float(x)
    except ValueError:
        return x


def parse_csv_values(s: str) -> List[Any]:
    return [parse_scalar(x) for x in s.split(",")]


def parse_refs(s: str) -> List[str]:
    return re.findall(r"'([^']+)'", s)


def parse_assistant_output(text: str) -> Dict[str, Any]:
    primitives: List[Dict[str, Any]] = []

    for primitive_id, values in LINE_RE.findall(text):
        parsed = parse_csv_values(values)
        if len(parsed) != 5:
            continue
        x1, y1, x2, y2, is_valid = parsed
        primitives.append(
            {
                "id": primitive_id,
                "type": "line",
                "geometry": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
                "is_valid": bool(is_valid),
            }
        )

    for primitive_id, values in CIRCLE_RE.findall(text):
        parsed = parse_csv_values(values)
        if len(parsed) != 3:
            continue
        cx, cy, r = parsed
        primitives.append(
            {
                "id": primitive_id,
                "type": "circle",
                "geometry": {"cx": float(cx), "cy": float(cy), "r": float(r)},
            }
        )

    for primitive_id, values in ARC_RE.findall(text):
        parsed = parse_csv_values(values)
        if len(parsed) != 5:
            continue
        cx, cy, r, start_param, end_param = parsed
        primitives.append(
            {
                "id": primitive_id,
                "type": "arc",
                "geometry": {
                    "cx": float(cx),
                    "cy": float(cy),
                    "r": float(r),
                    "start_param": float(start_param),
                    "end_param": float(end_param),
                },
            }
        )

    constraints: List[Dict[str, Any]] = []
    for ctype, source, target, p1, p2 in CONSTRAINT_RE.findall(text):
        constraints.append(
            {
                "type": ctype,
                "source": parse_scalar(source),
                "target": parse_scalar(target),
                "pointType1": parse_scalar(p1),
                "pointType2": parse_scalar(p2),
            }
        )

    dimensions: List[Dict[str, Any]] = []
    for dtype, refs, value in DIMENSION_RE.findall(text):
        dimensions.append(
            {
                "type": dtype.lower(),
                "refs": parse_refs(refs),
                "value": float(value),
                "unit": "unknown",
            }
        )

    return {
        "primitives": primitives,
        "constraints": constraints,
        "dimensions": dimensions,
    }


def primitive_id_from_ref(ref: str) -> Optional[str]:
    if "." not in ref:
        return None
    return ref.split(".", 1)[0]


def add_dimension_associations(dimensions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(dimensions):
        d = dict(d)
        d["id"] = f"dim_{i}"
        d["associated_primitives"] = sorted(
            {
                primitive_id_from_ref(ref)
                for ref in d.get("refs", [])
                if primitive_id_from_ref(ref) is not None
            }
        )
        out.append(d)
    return out


def point_for_ref(ref: str, primitive_index: Dict[str, Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if "." not in ref:
        return None

    primitive_id, ref_type_str = ref.split(".", 1)
    try:
        ref_type = int(ref_type_str)
    except ValueError:
        return None

    p = primitive_index.get(primitive_id)
    if not p:
        return None

    g = p["geometry"]

    if p["type"] == "line":
        if ref_type == 1:
            return g["x1"], g["y1"]
        if ref_type == 2:
            return g["x2"], g["y2"]

    if p["type"] == "circle":
        if ref_type == 3:
            return g["cx"], g["cy"]

    if p["type"] == "arc":
        if ref_type == 3:
            return g["cx"], g["cy"]
        if ref_type in {1, 2}:
            angle_deg = g["start_param"] if ref_type == 1 else g["end_param"]
            theta = math.radians(angle_deg)
            return (
                g["cx"] + g["r"] * math.cos(theta),
                g["cy"] + g["r"] * math.sin(theta),
            )

    return None


def add_geometry_checks(record: Dict[str, Any]) -> Dict[str, Any]:
    primitive_index = {p["id"]: p for p in record["primitives"]}

    for d in record["dimensions"]:
        refs = d.get("refs", [])
        dtype = d.get("type")
        d["computed_from_geometry"] = None

        try:
            if dtype == "linear" and len(refs) == 2:
                p1 = point_for_ref(refs[0], primitive_index)
                p2 = point_for_ref(refs[1], primitive_index)
                if p1 and p2:
                    d["computed_from_geometry"] = math.dist(p1, p2)

            elif dtype == "diameter" and len(refs) == 1:
                primitive_id = primitive_id_from_ref(refs[0])
                p = primitive_index.get(primitive_id) if primitive_id else None
                if p and p["type"] in {"circle", "arc"}:
                    d["computed_from_geometry"] = 2.0 * p["geometry"]["r"]

            elif dtype == "radius" and len(refs) == 1:
                primitive_id = primitive_id_from_ref(refs[0])
                p = primitive_index.get(primitive_id) if primitive_id else None
                if p and p["type"] in {"circle", "arc"}:
                    d["computed_from_geometry"] = p["geometry"]["r"]

            elif dtype == "angular" and refs:
                primitive_id = primitive_id_from_ref(refs[0])
                p = primitive_index.get(primitive_id) if primitive_id else None
                if p and p["type"] == "arc":
                    start = p["geometry"]["start_param"]
                    end = p["geometry"]["end_param"]
                    d["computed_from_geometry"] = abs(end - start)

        except Exception:
            d["computed_from_geometry"] = None

    return record


def normalize_primitives(primitives: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    if width <= 0 or height <= 0:
        return primitives

    out: List[Dict[str, Any]] = []
    radius_scale = max(width, height)

    for p in primitives:
        p2 = {
            k: (dict(v) if k == "geometry" and isinstance(v, dict) else v)
            for k, v in p.items()
        }
        g = p2["geometry"]

        if p2["type"] == "line":
            g["x1"] /= width
            g["x2"] /= width
            g["y1"] /= height
            g["y2"] /= height
        elif p2["type"] in {"circle", "arc"}:
            g["cx"] /= width
            g["cy"] /= height
            g["r"] /= radius_scale

        out.append(p2)

    return out


def extract_label_text(item: Dict[str, Any]) -> Optional[str]:
    conversations = item.get("conversations")
    if isinstance(conversations, list):
        for msg in conversations:
            if not isinstance(msg, dict):
                continue
            if msg.get("from") in {"gpt", "assistant"} and isinstance(msg.get("value"), str):
                return msg["value"]

    for key in ("label", "labels", "output", "answer", "value"):
        if isinstance(item.get(key), str):
            return item[key]

    return None


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def stable_split(record_key: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.blake2b(record_key.encode("utf-8", errors="ignore"), digest_size=8).digest()
    x = int.from_bytes(digest, "big") / float(1 << 64)
    if x < train_ratio:
        return "train"
    if x < train_ratio + val_ratio:
        return "val"
    return "test"


def iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json_loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except Exception:
                    continue
        return

    if ijson is not None:
        try:
            with path.open("rb") as f:
                first = f.read(1)
                f.seek(0)
                if first == b"[":
                    for obj in ijson.items(f, "item"):
                        if isinstance(obj, dict):
                            yield obj
                    return
        except Exception:
            pass

    text = path.read_bytes()
    if not text.strip():
        return

    try:
        obj = json_loads(text)
    except Exception:
        # Last-chance JSONL fallback for mislabeled files.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json_loads(line)
                if isinstance(item, dict):
                    yield item
            except Exception:
                continue
        return

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        yield obj


def fast_image_size(path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        with path.open("rb") as f:
            head = f.read(32)

            if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
                width, height = struct.unpack(">II", head[16:24])
                return int(width), int(height)

            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    marker_start = f.read(1)
                    if not marker_start:
                        break
                    if marker_start != b"\xff":
                        continue
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if not marker:
                        break
                    marker_int = marker[0]
                    if marker_int in {0xD8, 0xD9}:
                        continue
                    size_bytes = f.read(2)
                    if len(size_bytes) != 2:
                        break
                    segment_size = struct.unpack(">H", size_bytes)[0]
                    if segment_size < 2:
                        break
                    if marker_int in {
                        0xC0,
                        0xC1,
                        0xC2,
                        0xC3,
                        0xC5,
                        0xC6,
                        0xC7,
                        0xC9,
                        0xCA,
                        0xCB,
                        0xCD,
                        0xCE,
                        0xCF,
                    }:
                        data = f.read(5)
                        if len(data) == 5:
                            height, width = struct.unpack(">HH", data[1:5])
                            return int(width), int(height)
                        break
                    f.seek(segment_size - 2, os.SEEK_CUR)

            if head.startswith(b"BM") and len(head) >= 26:
                width = struct.unpack("<I", head[18:22])[0]
                height = abs(struct.unpack("<i", head[22:26])[0])
                return int(width), int(height)

            if head.startswith((b"GIF87a", b"GIF89a")) and len(head) >= 10:
                width, height = struct.unpack("<HH", head[6:10])
                return int(width), int(height)

            if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
                # Fall through to Pillow for WebP. The header variants are more annoying
                # than they are worth here, and WebP is rarely the hot path for ParaCAD.
                pass

    except Exception:
        return None, None

    if Image is not None:
        try:
            with Image.open(path) as img:
                return img.width, img.height
        except Exception:
            return None, None

    return None, None


def resolve_image_path(image_name: str) -> Optional[Path]:
    p = Path(image_name)

    if p.is_absolute() and p.exists():
        return p

    for root_str in G_SEARCH_ROOTS:
        candidate = Path(root_str) / p
        if candidate.exists():
            return candidate

    indexed = G_IMAGE_INDEX.get(p.name)
    if indexed:
        candidate = Path(indexed)
        if candidate.exists():
            return candidate

    return None


def process_item(item: Dict[str, Any], json_path: Path, source_index: int) -> Tuple[Optional[Dict[str, Any]], str]:
    image_name = item.get("image") or item.get("image_path") or item.get("file_name")
    label_text = extract_label_text(item)

    if not image_name or not isinstance(image_name, str):
        return None, "missing_image"
    if not label_text:
        return None, "missing_label"

    parsed = parse_assistant_output(label_text)
    if not parsed["primitives"]:
        return None, "missing_primitives"

    image_path = resolve_image_path(image_name)
    width: Optional[int] = None
    height: Optional[int] = None

    if image_path is not None and G_READ_IMAGE_SIZES:
        width, height = fast_image_size(image_path)

    primitives = parsed["primitives"]
    if G_NORMALIZE:
        if width and height:
            primitives = normalize_primitives(primitives, width, height)
        else:
            return None, "missing_image_size_for_normalize"

    dimensions = add_dimension_associations(parsed["dimensions"])

    source_root = Path(G_SOURCE_ROOT)
    record = {
        "source_json": safe_rel(json_path, source_root),
        "source_index": source_index,
        "image": image_name,
        "image_path": safe_rel(image_path, source_root) if image_path else None,
        "image_width": width,
        "image_height": height,
        "primitives": primitives,
        "constraints": parsed["constraints"],
        "dimensions": dimensions,
    }

    return add_geometry_checks(record), "ok"


def init_worker(
    image_index: Dict[str, str],
    search_roots: List[str],
    source_root: str,
    output_root: str,
    shard_root: str,
    normalize: bool,
    read_image_sizes: bool,
    train_ratio: float,
    val_ratio: float,
    write_all: bool,
) -> None:
    global G_IMAGE_INDEX
    global G_SEARCH_ROOTS
    global G_SOURCE_ROOT
    global G_OUTPUT_ROOT
    global G_SHARD_ROOT
    global G_NORMALIZE
    global G_READ_IMAGE_SIZES
    global G_TRAIN_RATIO
    global G_VAL_RATIO
    global G_WRITE_ALL

    G_IMAGE_INDEX = image_index
    G_SEARCH_ROOTS = search_roots
    G_SOURCE_ROOT = source_root
    G_OUTPUT_ROOT = output_root
    G_SHARD_ROOT = shard_root
    G_NORMALIZE = normalize
    G_READ_IMAGE_SIZES = read_image_sizes
    G_TRAIN_RATIO = train_ratio
    G_VAL_RATIO = val_ratio
    G_WRITE_ALL = write_all


def process_json_file_task(task: Tuple[int, str]) -> Dict[str, Any]:
    task_id, json_path_str = task
    json_path = Path(json_path_str)
    stats: Counter[str] = Counter()
    shard_root = Path(G_SHARD_ROOT)

    shard_paths = {
        "train": shard_root / f"train_{task_id:08d}.jsonl",
        "val": shard_root / f"val_{task_id:08d}.jsonl",
        "test": shard_root / f"test_{task_id:08d}.jsonl",
    }
    if G_WRITE_ALL:
        shard_paths["all"] = shard_root / f"all_{task_id:08d}.jsonl"

    handles = {name: path.open("wb") for name, path in shard_paths.items()}

    try:
        for source_index, item in enumerate(iter_json_records(json_path)):
            stats["raw_records"] += 1
            record, status = process_item(item, json_path, source_index)
            stats[status] += 1
            if record is None:
                continue

            key = f"{record.get('image','')}|{record.get('source_json','')}|{record.get('source_index',0)}"
            split = stable_split(key, G_TRAIN_RATIO, G_VAL_RATIO)
            line = json_dumps_line(record)
            handles[split].write(line)
            if G_WRITE_ALL:
                handles["all"].write(line)

    except Exception as exc:
        stats["file_errors"] += 1
        return {
            "task_id": task_id,
            "json_path": json_path_str,
            "ok": False,
            "error": repr(exc),
            "stats": dict(stats),
            "shards": {k: str(v) for k, v in shard_paths.items()},
        }
    finally:
        for handle in handles.values():
            handle.close()

    return {
        "task_id": task_id,
        "json_path": json_path_str,
        "ok": True,
        "error": None,
        "stats": dict(stats),
        "shards": {k: str(v) for k, v in shard_paths.items()},
    }


def path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def iter_files(root: Path, exts: set[str], skip_dirs: Sequence[Path] = ()) -> Iterator[Path]:
    skip_resolved = [p.resolve() for p in skip_dirs if p.exists()]
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        if any(path_is_under(current, skip) for skip in skip_resolved):
            dirnames[:] = []
            continue

        for filename in filenames:
            p = current / filename
            if p.suffix.lower() in exts:
                yield p


def scan_files(search_roots: Sequence[Path], output_root: Path) -> Tuple[List[Path], List[Path]]:
    json_files: List[Path] = []
    image_files: List[Path] = []

    for root in search_roots:
        json_files.extend(iter_files(root, JSON_EXTS, skip_dirs=[output_root]))
        image_files.extend(iter_files(root, IMAGE_EXTS, skip_dirs=[output_root]))

    return sorted(set(json_files)), sorted(set(image_files))


def build_image_index(
    image_files: Sequence[Path], progress: Optional[RichProgress] = None, task_id: Optional[int] = None
) -> Tuple[Dict[str, str], int]:
    index: Dict[str, str] = {}
    duplicate_names = 0

    for path in image_files:
        name = path.name
        if name in index:
            duplicate_names += 1
        else:
            index[name] = str(path)
        if progress is not None:
            progress.advance(task_id)

    return index, duplicate_names


def validate_zip_members(zip_path: Path, target: Path) -> None:
    target_resolved = target.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            destination = (target / member.filename).resolve()
            if destination != target_resolved and target_resolved not in destination.parents:
                raise ValueError(f"Unsafe zip member path: {member.filename}")


def extract_one_zip(job: Tuple[str, str, bool]) -> Dict[str, Any]:
    zip_path = Path(job[0])
    target = Path(job[1])
    overwrite = job[2]

    try:
        if target.exists() and overwrite:
            shutil.rmtree(target)

        if target.exists() and any(target.iterdir()):
            return {"ok": True, "zip": str(zip_path), "status": "already_extracted"}

        target.mkdir(parents=True, exist_ok=True)
        validate_zip_members(zip_path, target)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target)

        return {"ok": True, "zip": str(zip_path), "status": "extracted"}
    except zipfile.BadZipFile:
        return {"ok": False, "zip": str(zip_path), "status": "bad_zip"}
    except Exception as exc:
        return {"ok": False, "zip": str(zip_path), "status": repr(exc)}


def extract_zips_parallel(root: Path, extract_dir: Path, workers: int, overwrite: bool, show_progress: bool = True) -> Counter[str]:
    zip_paths = sorted(iter_files(root, {".zip"}, skip_dirs=[extract_dir]))
    extract_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    if not zip_paths:
        return stats

    jobs = []
    for zip_path in zip_paths:
        try:
            rel = str(zip_path.resolve().relative_to(root.resolve()))
        except Exception:
            rel = str(zip_path)
        tag = hashlib.blake2b(rel.encode("utf-8", errors="ignore"), digest_size=5).hexdigest()
        target = extract_dir / f"{zip_path.with_suffix('').name}_{tag}"
        jobs.append((str(zip_path), str(target), overwrite))

    print(f"[ZIP] Found {len(jobs):,} zip files; extracting with {workers} worker(s)")
    with RichProgress(enabled=show_progress) as progress:
        task = progress.add_task("Extracting ZIP archives", total=len(jobs))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(extract_one_zip, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                stats[result["status"]] += 1
                if task is not None:
                    progress.update(task, completed=i, description=f"Extracting ZIPs: {dict(stats)}")
                elif i % 25 == 0 or i == len(futures):
                    print(f"[ZIP] {i:,}/{len(futures):,} done | {dict(stats)}")
        progress.complete(task, "ZIP extraction complete")

    return stats


def merge_shards(shard_root: Path, output_root: Path, names: Sequence[str], show_progress: bool = True) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with RichProgress(enabled=show_progress) as progress:
        task = progress.add_task("Merging output shards", total=len(names))
        for name in names:
            output_path = output_root / f"{name}.jsonl"
            shard_paths = sorted(shard_root.glob(f"{name}_*.jsonl"))
            line_count = 0

            with output_path.open("wb") as out:
                for shard_path in shard_paths:
                    with shard_path.open("rb") as inp:
                        shutil.copyfileobj(inp, out, length=16 * 1024 * 1024)
                    try:
                        with shard_path.open("rb") as f:
                            line_count += sum(1 for _ in f)
                    except Exception:
                        pass

            counts[name] = line_count
            progress.advance(task)
            print(f"[MERGE] {name}.jsonl <- {len(shard_paths):,} shards, {line_count:,} records")
        progress.complete(task, "Shard merge complete")

    return counts


def remove_old_outputs(output_root: Path, keep_extracted: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "all.jsonl", "manifest.json"):
        p = output_root / name
        if p.exists():
            p.unlink()

    shard_root = output_root / "_shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)

    if not keep_extracted:
        extracted_root = output_root / "_extracted_zips"
        if extracted_root.exists():
            shutil.rmtree(extracted_root)


def write_manifest(output_root: Path, manifest: Dict[str, Any]) -> None:
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def positive_ratio(value: str) -> float:
    x = float(value)
    if not 0 <= x <= 1:
        raise argparse.ArgumentTypeError("ratio must be between 0 and 1")
    return x


def parse_args() -> argparse.Namespace:
    cpu_count = os.cpu_count() or 8
    default_workers = max(1, cpu_count - 2)

    parser = argparse.ArgumentParser(
        description="Parallel ParaCAD preprocessor for JSON/image/zip datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Folder containing ParaCAD files.")
    parser.add_argument("--output", required=True, help="Output folder for processed JSONL files.")
    parser.add_argument("--extract-zips", action="store_true", help="Extract zip files before processing.")
    parser.add_argument("--overwrite-extracted", action="store_true", help="Re-extract zip folders from scratch.")
    parser.add_argument("--normalize", action="store_true", help="Normalize primitive coordinates to [0, 1].")
    parser.add_argument("--read-image-sizes", action="store_true", help="Store image width/height even without --normalize.")
    parser.add_argument("--no-image-index", action="store_true", help="Skip filename image index for faster startup.")
    parser.add_argument("--no-all", action="store_true", help="Do not write all.jsonl, reducing disk writes.")
    parser.add_argument("--keep-shards", action="store_true", help="Keep temporary worker shards after merge.")
    parser.add_argument("--workers", type=int, default=default_workers, help="JSON parsing process workers.")
    parser.add_argument("--zip-workers", type=int, default=3, help="Zip extraction thread workers.")
    parser.add_argument("--train-ratio", type=positive_ratio, default=0.90)
    parser.add_argument("--val-ratio", type=positive_ratio, default=0.05)
    parser.add_argument("--seed", type=int, default=42, help="Used only for task ordering, not split assignment.")
    parser.add_argument("--limit-json-files", type=int, default=0, help="Debug: process only the first N JSON files.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress after this many JSON files.")
    add_progress_argument(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.train_ratio + args.val_ratio > 1.0:
        print("[ERROR] --train-ratio + --val-ratio must be <= 1.0", file=sys.stderr)
        return 2

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    extracted_root = output_root / "_extracted_zips"
    shard_root = output_root / "_shards"

    if not input_root.exists():
        print(f"[ERROR] input folder does not exist: {input_root}", file=sys.stderr)
        return 2

    start_time = time.time()
    remove_old_outputs(output_root, keep_extracted=args.extract_zips and not args.overwrite_extracted)
    shard_root.mkdir(parents=True, exist_ok=True)

    zip_stats: Counter[str] = Counter()
    if args.extract_zips:
        zip_stats = extract_zips_parallel(
            root=input_root,
            extract_dir=extracted_root,
            workers=args.zip_workers,
            overwrite=args.overwrite_extracted,
            show_progress=not args.no_progress,
        )

    search_roots = [input_root]
    if extracted_root.exists():
        search_roots.append(extracted_root)

    print("[SCAN] Finding JSON and image files")
    with RichProgress(enabled=not args.no_progress) as progress:
        scan_task = progress.add_task("Scanning ParaCAD input files", total=2)
        json_files, image_files = scan_files(search_roots, output_root)
        progress.advance(scan_task)
        if args.limit_json_files > 0:
            json_files = json_files[: args.limit_json_files]
        progress.complete(scan_task, "Input file scan complete")

    image_index: Dict[str, str] = {}
    duplicate_image_names = 0
    if not args.no_image_index:
        print(f"[SCAN] Building image filename index for {len(image_files):,} images")
        with RichProgress(enabled=not args.no_progress) as progress:
            index_task = progress.add_task("Indexing image filenames", total=len(image_files))
            image_index, duplicate_image_names = build_image_index(image_files, progress, index_task)
            progress.complete(index_task, "Image filename index complete")
    else:
        print("[SCAN] Skipping image filename index")

    print(f"[SCAN] JSON files: {len(json_files):,}")
    print(f"[SCAN] Image files: {len(image_files):,}")
    print(f"[SCAN] Indexed image names: {len(image_index):,}")
    if duplicate_image_names:
        print(f"[SCAN] Duplicate image basenames ignored: {duplicate_image_names:,}")

    if not json_files:
        print("[DONE] No JSON/JSONL files found")
        write_manifest(
            output_root,
            {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "num_json_files_found": 0,
                "num_images_found": len(image_files),
                "num_records": 0,
            },
        )
        return 0

    random.seed(args.seed)
    random.shuffle(json_files)

    read_image_sizes = bool(args.normalize or args.read_image_sizes)
    write_all = not args.no_all
    task_stats: Counter[str] = Counter()
    failed_files: List[Dict[str, Any]] = []
    tasks = [(i, str(path)) for i, path in enumerate(json_files)]

    print(f"[WORK] Processing with {args.workers} process worker(s)")
    with RichProgress(enabled=not args.no_progress) as progress:
        work_task = progress.add_task("Processing ParaCAD JSON files", total=len(tasks))
        with ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            initializer=init_worker,
            initargs=(
                image_index,
                [str(p) for p in search_roots],
                str(input_root),
                str(output_root),
                str(shard_root),
                bool(args.normalize),
                read_image_sizes,
                float(args.train_ratio),
                float(args.val_ratio),
                write_all,
            ),
        ) as ex:
            futures = [ex.submit(process_json_file_task, task) for task in tasks]

            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                task_stats.update(result.get("stats", {}))
                if not result.get("ok"):
                    failed_files.append(
                        {
                            "json_path": result.get("json_path"),
                            "error": result.get("error"),
                            "stats": result.get("stats", {}),
                        }
                    )
                if work_task is not None:
                    progress.update(work_task, completed=done, description=f"Processing: {task_stats.get('ok', 0):,} records")
                elif done % args.progress_every == 0 or done == len(futures):
                    elapsed = max(1e-6, time.time() - start_time)
                    files_per_sec = done / elapsed
                    ok_records = task_stats.get("ok", 0)
                    print(
                        f"[WORK] {done:,}/{len(futures):,} files | "
                        f"{ok_records:,} records | {files_per_sec:.2f} files/s"
                    )
        progress.complete(work_task, f"JSON processing complete: {task_stats.get('ok', 0):,} records")

    names = ["train", "val", "test"]
    if write_all:
        names.append("all")

    print("[MERGE] Combining worker shards")
    split_counts = merge_shards(shard_root, output_root, names, show_progress=not args.no_progress)

    if not args.keep_shards:
        shutil.rmtree(shard_root, ignore_errors=True)

    elapsed = time.time() - start_time
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "elapsed_seconds": round(elapsed, 3),
        "workers": args.workers,
        "zip_workers": args.zip_workers,
        "extract_zips": bool(args.extract_zips),
        "zip_stats": dict(zip_stats),
        "num_json_files_found": len(json_files),
        "num_images_found": len(image_files),
        "num_indexed_image_names": len(image_index),
        "duplicate_image_basenames_ignored": duplicate_image_names,
        "normalized_coordinates": bool(args.normalize),
        "read_image_sizes": read_image_sizes,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
        "split_counts": split_counts,
        "record_stats": dict(task_stats),
        "failed_files": failed_files[:100],
        "num_failed_files": len(failed_files),
        "used_orjson": orjson is not None,
        "used_ijson": ijson is not None,
        "used_pillow": Image is not None,
    }
    write_manifest(output_root, manifest)

    print("[DONE]")
    print(json.dumps(manifest, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
