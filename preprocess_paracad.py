# preprocess_paracad.py
import argparse
import json
import math
import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cli_progress import RichProgress, add_progress_argument

try:
    from PIL import Image
except ImportError:
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


def parse_scalar(x: str) -> Any:
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


def safe_json_load(path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
    1. Normal JSON array: [{...}, {...}]
    2. Single JSON object: {...}
    3. JSONL: one JSON object per line
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()

    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        return []
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                pass
        return records


def parse_assistant_output(text: str) -> Dict[str, Any]:
    primitives = []

    for primitive_id, values in LINE_RE.findall(text):
        parsed = parse_csv_values(values)
        if len(parsed) != 5:
            continue

        x1, y1, x2, y2, is_valid = parsed
        primitives.append({
            "id": primitive_id,
            "type": "line",
            "geometry": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            },
            "is_valid": bool(is_valid),
        })

    for primitive_id, values in CIRCLE_RE.findall(text):
        parsed = parse_csv_values(values)
        if len(parsed) != 3:
            continue

        cx, cy, r = parsed
        primitives.append({
            "id": primitive_id,
            "type": "circle",
            "geometry": {
                "cx": float(cx),
                "cy": float(cy),
                "r": float(r),
            },
        })

    for primitive_id, values in ARC_RE.findall(text):
        parsed = parse_csv_values(values)
        if len(parsed) != 5:
            continue

        cx, cy, r, start_param, end_param = parsed
        primitives.append({
            "id": primitive_id,
            "type": "arc",
            "geometry": {
                "cx": float(cx),
                "cy": float(cy),
                "r": float(r),
                "start_param": float(start_param),
                "end_param": float(end_param),
            },
        })

    constraints = []
    for ctype, source, target, p1, p2 in CONSTRAINT_RE.findall(text):
        constraints.append({
            "type": ctype,
            "source": parse_scalar(source),
            "target": parse_scalar(target),
            "pointType1": parse_scalar(p1),
            "pointType2": parse_scalar(p2),
        })

    dimensions = []
    for dtype, refs, value in DIMENSION_RE.findall(text):
        dimensions.append({
            "type": dtype.lower(),
            "refs": parse_refs(refs),
            "value": float(value),
            "unit": "unknown",
        })

    return {
        "primitives": primitives,
        "constraints": constraints,
        "dimensions": dimensions,
    }


def extract_zips(root: Path, extract_dir: Path, delete_existing: bool = False, show_progress: bool = True) -> None:
    zip_paths = list(root.rglob("*.zip"))

    if delete_existing and extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    with RichProgress(enabled=show_progress) as progress:
        task = progress.add_task("Extracting ZIP archives", total=len(zip_paths))
        for zip_path in zip_paths:
            target = extract_dir / zip_path.stem

            if target.exists() and any(target.iterdir()):
                print(f"[ZIP] already extracted: {zip_path.name}")
                progress.advance(task)
                continue

            target.mkdir(parents=True, exist_ok=True)
            print(f"[ZIP] extracting: {zip_path} -> {target}")

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(target)
            except zipfile.BadZipFile:
                print(f"[WARN] bad zip skipped: {zip_path}")
            progress.advance(task)
        progress.complete(task, "ZIP extraction complete")


def build_image_index(root: Path, show_progress: bool = True) -> Dict[str, Path]:
    """
    Maps image filename -> full path.
    If duplicates exist, the first one found is used.
    """
    index = {}

    with RichProgress(enabled=show_progress) as progress:
        task = progress.add_task("Building image filename index")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                index.setdefault(path.name, path)
                progress.advance(task)
        progress.complete(task, f"Image index complete: {len(index):,} names")

    return index


def get_image_size(path: Optional[Path]) -> Tuple[Optional[int], Optional[int]]:
    if path is None or not path.exists() or Image is None:
        return None, None

    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def normalize_primitives(primitives: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    if not width or not height:
        return primitives

    out = []

    for p in primitives:
        p = json.loads(json.dumps(p))
        g = p["geometry"]

        if p["type"] == "line":
            g["x1"] /= width
            g["x2"] /= width
            g["y1"] /= height
            g["y2"] /= height

        elif p["type"] in {"circle", "arc"}:
            g["cx"] /= width
            g["cy"] /= height
            # Radius is ambiguous if width != height.
            # Use max dimension so the model gets stable normalized scale.
            g["r"] /= max(width, height)

        out.append(p)

    return out


def primitive_id_from_ref(ref: str) -> Optional[str]:
    if "." not in ref:
        return None
    return ref.split(".", 1)[0]


def add_dimension_associations(dimensions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

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
    """
    Resolves refs like:
      Line4.1 -> line start point
      Line4.2 -> line end point
      Circle0.3 -> circle center
      Arc0.1 -> arc start point
      Arc0.2 -> arc end point
      Arc0.3 -> arc center
    """
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
    """
    Adds optional geometric values computed from the coordinate data.
    Do not assume these match printed dimensions exactly; ParaCAD values may be CAD-space,
    while coordinates are image/render-space.
    """
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
                p = primitive_index.get(primitive_id)

                if p and p["type"] in {"circle", "arc"}:
                    d["computed_from_geometry"] = 2.0 * p["geometry"]["r"]

            elif dtype == "radius" and len(refs) == 1:
                primitive_id = primitive_id_from_ref(refs[0])
                p = primitive_index.get(primitive_id)

                if p and p["type"] in {"circle", "arc"}:
                    d["computed_from_geometry"] = p["geometry"]["r"]

            elif dtype == "angular" and len(refs) >= 1:
                primitive_id = primitive_id_from_ref(refs[0])
                p = primitive_index.get(primitive_id)

                if p and p["type"] == "arc":
                    start = p["geometry"]["start_param"]
                    end = p["geometry"]["end_param"]
                    d["computed_from_geometry"] = abs(end - start)

        except Exception:
            d["computed_from_geometry"] = None

    return record


def extract_label_text(item: Dict[str, Any]) -> Optional[str]:
    """
    Handles ParaCAD conversation format:
      conversations[1]["from"] == "gpt"
      conversations[1]["value"] == label text
    """
    conversations = item.get("conversations")

    if isinstance(conversations, list):
        for msg in conversations:
            if msg.get("from") in {"gpt", "assistant"} and isinstance(msg.get("value"), str):
                return msg["value"]

    # Fallbacks for already-processed formats.
    for key in ["label", "labels", "output", "answer", "value"]:
        if isinstance(item.get(key), str):
            return item[key]

    return None


def process_json_file(
    json_path: Path,
    image_index: Dict[str, Path],
    normalize: bool,
    source_root: Path,
) -> List[Dict[str, Any]]:
    raw_records = safe_json_load(json_path)
    processed = []

    for idx, item in enumerate(raw_records):
        image_name = item.get("image") or item.get("image_path") or item.get("file_name")
        label_text = extract_label_text(item)

        if not image_name or not label_text:
            continue

        parsed = parse_assistant_output(label_text)

        if not parsed["primitives"]:
            continue

        image_path = image_index.get(Path(image_name).name)
        width, height = get_image_size(image_path)

        primitives = parsed["primitives"]
        if normalize and width and height:
            primitives = normalize_primitives(primitives, width, height)

        dimensions = add_dimension_associations(parsed["dimensions"])

        record = {
            "source_json": str(json_path.relative_to(source_root)),
            "source_index": idx,
            "image": image_name,
            "image_path": str(image_path.relative_to(source_root)) if image_path else None,
            "image_width": width,
            "image_height": height,
            "primitives": primitives,
            "constraints": parsed["constraints"],
            "dimensions": dimensions,
        }

        record = add_geometry_checks(record)
        processed.append(record)

    return processed


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing ParaCAD files, images, json, and zips.")
    parser.add_argument("--output", required=True, help="Output folder for processed dataset.")
    parser.add_argument("--extract-zips", action="store_true", help="Extract zip files before processing.")
    parser.add_argument("--normalize", action="store_true", help="Normalize coordinates to [0, 1].")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    add_progress_argument(parser)
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    extracted_root = output_root / "_extracted_zips"

    if args.extract_zips:
        extract_zips(input_root, extracted_root, show_progress=not args.no_progress)

    search_roots = [input_root]
    if extracted_root.exists():
        search_roots.append(extracted_root)

    print("[SCAN] Building image index...")
    image_index = {}
    for root in search_roots:
        image_index.update(build_image_index(root, show_progress=not args.no_progress))

    print(f"[SCAN] Found {len(image_index):,} image files")

    json_files = []
    for root in search_roots:
        json_files.extend(root.rglob("*.json"))
        json_files.extend(root.rglob("*.jsonl"))

    json_files = sorted(set(json_files))
    print(f"[SCAN] Found {len(json_files):,} JSON/JSONL files")

    all_records = []
    skipped_files = 0

    with RichProgress(enabled=not args.no_progress) as progress:
        task = progress.add_task("Processing ParaCAD JSON files", total=len(json_files))
        for json_path in json_files:
            try:
                records = process_json_file(
                    json_path=json_path,
                    image_index=image_index,
                    normalize=args.normalize,
                    source_root=input_root,
                )
                all_records.extend(records)
            except Exception as e:
                skipped_files += 1
                print(f"[WARN] skipped {json_path}: {e}")
            progress.advance(task)
        progress.complete(task, f"JSON processing complete: {len(all_records):,} records")

    random.seed(args.seed)
    random.shuffle(all_records)

    n = len(all_records)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train = all_records[:n_train]
    val = all_records[n_train:n_train + n_val]
    test = all_records[n_train + n_val:]

    write_jsonl(output_root / "train.jsonl", train)
    write_jsonl(output_root / "val.jsonl", val)
    write_jsonl(output_root / "test.jsonl", test)
    write_jsonl(output_root / "all.jsonl", all_records)

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_images_found": len(image_index),
        "num_json_files_found": len(json_files),
        "num_records": n,
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
        "skipped_files": skipped_files,
        "normalized_coordinates": args.normalize,
    }

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
