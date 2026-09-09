#!/usr/bin/env python3
"""Train or run a ParaCAD image-to-primitive DETR model.

The model takes a raster engineering drawing and predicts a *set* of CAD
primitives: lines, circles, and arcs.  It uses the primitive tables in
``ParaCAD_full_v3.zarr`` as supervision and is deliberately independent of
the label JSON blobs.

This is a geometry-recognition model.  Constraints and dimensions are not
predicted by this first stage; they should be inferred after geometry is
recovered or learned by a second graph model.

Quick start:

    python -m pip install -r requirements-primitive-detr.txt
    python paracad_primitive_detr.py train --dataset ParaCAD_full_v3.zarr \
        --epochs 30 --train-max-samples 0 --val-max-samples 0

    python paracad_primitive_detr.py predict --checkpoint checkpoints/best.pt \
        --image drawing.png --output predicted_primitives.json

The defaults are intentionally a smaller smoke-training run (100,000 samples)
instead of an accidental multi-day full-dataset training job.  Pass
``--train-max-samples 0`` to train with every sample in the selected split.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from cli_progress import RichProgress, add_progress_argument

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - optional optimized matcher
    linear_sum_assignment = None

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - dependency guard
    print(
        "[ERROR] Missing model dependencies. Install them with:\n"
        "  python -m pip install -r requirements-primitive-detr.txt\n"
        f"Import error: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2)

# Leave several logical processors available for the OS, Streamlit, and the
# foreground training process.  This keeps Zarr/Pillow decoding ahead of a
# CUDA device without launching an unbounded number of workers.
DEFAULT_WORKERS = min(16, max((os.cpu_count() or 2) - 8, 2))


NO_OBJECT, LINE, CIRCLE, ARC = range(4)
CLASS_NAMES = {NO_OBJECT: "no_object", LINE: "line", CIRCLE: "circle", ARC: "arc"}
PRIMITIVE_TYPE_TO_CLASS = {1: LINE, 2: CIRCLE, 3: ARC}
GEOMETRY_SIZE = 11


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA-capable GPU.")
    return device


def configure_acceleration(device: torch.device) -> None:
    """Enable safe CUDA matmul and convolution fast paths for fixed-size inputs."""
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # TransformerDecoder uses scaled-dot-product attention on supported
    # PyTorch builds.  Explicitly enable the fused CUDA kernels when present.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def make_loader(
    dataset: Dataset[Any], batch_size: int, shuffle: bool, num_workers: int, device: torch.device
) -> DataLoader[Any]:
    """Build a prefetched, persistent DataLoader when worker processes are enabled."""
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_drawings,
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 4
    return DataLoader(dataset, **options)


def maybe_compile_for_evaluation(model: nn.Module, image_size: int, device: torch.device, enabled: bool) -> nn.Module:
    """Compile when supported, otherwise keep eager CUDA execution usable."""
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print("[WARN] --compile requires PyTorch 2.0+; using eager execution.", flush=True)
        return model
    if importlib.util.find_spec("triton") is None:
        print("[WARN] Triton is unavailable; skipping --compile and using eager CUDA execution.", flush=True)
        return model
    try:
        print("[INFO] Compiling the model for this evaluation run...", flush=True)
        compiled = torch.compile(model, mode="reduce-overhead")
        # Trigger compilation before the progress bar starts, so a compiler
        # problem becomes a clean fallback rather than ending evaluation later.
        with torch.inference_mode():
            compiled(torch.zeros((1, 1, image_size, image_size), device=device))
        return compiled
    except Exception as exc:  # pragma: no cover - depends on local compiler stack
        print(f"[WARN] torch.compile failed ({type(exc).__name__}); using eager CUDA execution.", flush=True)
        try:
            torch._dynamo.reset()  # type: ignore[attr-defined]
        except Exception:
            pass
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_to_tensor(image: Image.Image, image_size: int) -> tuple[Tensor, dict[str, float]]:
    """Letterbox a drawing and return its tensor plus label-coordinate transform."""
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    width, height = image.size
    scale = image_size / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas = Image.new("L", (image_size, image_size), color=255)
    canvas.paste(image.convert("L").resize((resized_width, resized_height), resampling), (left, top))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    transform = {
        "x_offset": left / image_size,
        "y_offset": top / image_size,
        "x_scale": resized_width / image_size,
        "y_scale": resized_height / image_size,
        "radius_scale": max(resized_width, resized_height) / image_size,
        "original_width": float(width),
        "original_height": float(height),
    }
    return torch.from_numpy((1.0 - pixels)[None]).contiguous(), transform


def encode_arc_angle(angle_degrees: float) -> tuple[float, float]:
    angle = math.radians(angle_degrees % 360.0)
    # Store sine/cosine in [0, 1], avoiding a discontinuity at 0/360 degrees.
    return (math.sin(angle) + 1.0) / 2.0, (math.cos(angle) + 1.0) / 2.0


def target_from_zarr_rows(
    type_ids: np.ndarray, geometry: np.ndarray, max_targets: int, transform: dict[str, float]
) -> dict[str, Tensor]:
    """Map ParaCAD's 7-value primitive table into masked model targets.

    The 11-value model representation is class-specific:
      line:   x1, y1, x2, y2
      circle: cx, cy, radius
      arc:    cx, cy, radius, sin(start), cos(start), sin(end), cos(end)
    """
    classes: list[int] = []
    values: list[list[float]] = []
    masks: list[list[float]] = []
    for primitive_type, row in zip(type_ids, geometry):
        output_class = PRIMITIVE_TYPE_TO_CLASS.get(int(primitive_type))
        if output_class is None:
            continue
        row = np.asarray(row, dtype=np.float32)
        value = [0.0] * GEOMETRY_SIZE
        mask = [0.0] * GEOMETRY_SIZE
        if output_class == LINE:
            if not np.isfinite(row[:4]).all():
                continue
            value[:4] = [
                transform["x_offset"] + float(np.clip(row[0], 0.0, 1.0)) * transform["x_scale"],
                transform["y_offset"] + float(np.clip(row[1], 0.0, 1.0)) * transform["y_scale"],
                transform["x_offset"] + float(np.clip(row[2], 0.0, 1.0)) * transform["x_scale"],
                transform["y_offset"] + float(np.clip(row[3], 0.0, 1.0)) * transform["y_scale"],
            ]
            mask[:4] = [1.0] * 4
        elif output_class == CIRCLE:
            if not np.isfinite(row[[0, 1, 4]]).all() or row[4] < 0:
                continue
            value[4:7] = [
                transform["x_offset"] + float(np.clip(row[0], 0.0, 1.0)) * transform["x_scale"],
                transform["y_offset"] + float(np.clip(row[1], 0.0, 1.0)) * transform["y_scale"],
                float(np.clip(row[4], 0.0, 1.0)) * transform["radius_scale"],
            ]
            mask[4:7] = [1.0] * 3
        else:  # ARC
            if not np.isfinite(row[[0, 1, 4, 5, 6]]).all() or row[4] < 0:
                continue
            start_sin, start_cos = encode_arc_angle(float(row[5]))
            end_sin, end_cos = encode_arc_angle(float(row[6]))
            value[4:11] = [
                transform["x_offset"] + float(np.clip(row[0], 0.0, 1.0)) * transform["x_scale"],
                transform["y_offset"] + float(np.clip(row[1], 0.0, 1.0)) * transform["y_scale"],
                float(np.clip(row[4], 0.0, 1.0)) * transform["radius_scale"],
                start_sin,
                start_cos,
                end_sin,
                end_cos,
            ]
            mask[4:11] = [1.0] * 7
        classes.append(output_class)
        values.append(value)
        masks.append(mask)
        if len(classes) == max_targets:
            break

    if values:
        geometry_tensor = torch.tensor(values, dtype=torch.float32)
        mask_tensor = torch.tensor(masks, dtype=torch.float32)
    else:
        geometry_tensor = torch.zeros((0, GEOMETRY_SIZE), dtype=torch.float32)
        mask_tensor = torch.zeros((0, GEOMETRY_SIZE), dtype=torch.float32)
    return {"classes": torch.tensor(classes, dtype=torch.long), "geometry": geometry_tensor, "geometry_mask": mask_tensor}


class ParaCADDrawingDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Lazily read image bytes and matching numeric labels from a Zarr archive."""

    def __init__(
        self,
        dataset_path: Path,
        split: str,
        image_size: int,
        max_targets: int,
        max_samples: int,
        seed: int,
    ) -> None:
        self.dataset_path = str(dataset_path)
        self.image_size = image_size
        self.max_targets = max_targets
        self._root: Any = None
        self._arrays: Optional[dict[str, Any]] = None
        try:
            import zarr  # type: ignore
        except ImportError as exc:
            raise RuntimeError('Missing dependency "zarr". Install requirements-primitive-detr.txt.') from exc

        root = zarr.open_group(self.dataset_path, mode="r")
        try:
            split_indices = np.asarray(root[f"splits/{split}"][:], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"Split '{split}' was not found in {dataset_path}.") from exc
        if max_samples > 0 and max_samples < split_indices.size:
            generator = np.random.default_rng(seed)
            selected = generator.choice(split_indices.size, size=max_samples, replace=False)
            split_indices = split_indices[selected]
        self.record_indices = split_indices
        # Keep only serializable state when a Windows DataLoader worker is spawned.
        self._root = None

    def __len__(self) -> int:
        return int(self.record_indices.size)

    def _open(self) -> dict[str, Any]:
        if self._arrays is not None:
            return self._arrays
        import zarr  # type: ignore

        self._root = zarr.open_group(self.dataset_path, mode="r")
        root = self._root
        self._arrays = {
            "image_id": root["records/image_id"],
            "primitive_offset": root["records/primitive_offset"],
            "primitive_count": root["records/primitive_count"],
            "image_bytes": root["images/data_bytes"],
            "image_offsets": root["images/data_offsets"],
            "primitive_type": root["primitives/type"],
            "primitive_geometry": root["primitives/geometry_norm"],
        }
        return self._arrays

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        arrays = self._open()
        record_index = int(self.record_indices[index])
        image_id = int(arrays["image_id"][record_index])
        if image_id < 0:
            raise RuntimeError(f"Record {record_index} has no linked image.")
        image_start, image_end = (int(value) for value in arrays["image_offsets"][image_id : image_id + 2])
        image_bytes = np.asarray(arrays["image_bytes"][image_start:image_end], dtype=np.uint8).tobytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            pixels, transform = image_to_tensor(image, self.image_size)

        primitive_start = int(arrays["primitive_offset"][record_index])
        primitive_count = int(arrays["primitive_count"][record_index])
        primitive_end = primitive_start + primitive_count
        type_ids = np.asarray(arrays["primitive_type"][primitive_start:primitive_end])
        geometry = np.asarray(arrays["primitive_geometry"][primitive_start:primitive_end])
        return pixels, target_from_zarr_rows(type_ids, geometry, self.max_targets, transform)


def collate_drawings(batch: Sequence[tuple[Tensor, dict[str, Tensor]]]) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        groups = min(32, output_channels)
        while output_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


def sine_position_encoding(height: int, width: int, channels: int, device: torch.device) -> Tensor:
    """Return (height * width, channels) fixed 2D position encoding."""
    if channels % 4:
        raise ValueError("d_model must be divisible by four for 2D sine positions.")
    feature_count = channels // 4
    y = torch.linspace(0.0, 1.0, height, device=device)
    x = torch.linspace(0.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    frequencies = 10000.0 ** (torch.arange(feature_count, device=device) / max(feature_count - 1, 1))
    x_encoding = grid_x[..., None] * (2.0 * math.pi) / frequencies
    y_encoding = grid_y[..., None] * (2.0 * math.pi) / frequencies
    position = torch.cat((x_encoding.sin(), x_encoding.cos(), y_encoding.sin(), y_encoding.cos()), dim=-1)
    return position.reshape(height * width, channels)


class PrimitiveDETR(nn.Module):
    """A compact DETR-style set predictor specialized for 2D CAD primitives."""

    def __init__(self, d_model: int = 256, num_queries: int = 128, decoder_layers: int = 6) -> None:
        super().__init__()
        if d_model % 8:
            raise ValueError("d_model must be divisible by eight.")
        self.d_model = d_model
        self.num_queries = num_queries
        self.backbone = nn.Sequential(
            ConvBlock(1, 64, 2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, d_model, 2),
            ConvBlock(d_model, d_model, 2),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers, norm=nn.LayerNorm(d_model))
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.geometry_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, GEOMETRY_SIZE)
        )

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.backbone(images)
        batch, channels, height, width = features.shape
        memory = features.flatten(2).transpose(1, 2)
        position = sine_position_encoding(height, width, channels, images.device).unsqueeze(0)
        memory = memory + position
        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return {"class_logits": self.class_head(decoded), "geometry": self.geometry_head(decoded).sigmoid()}


def gpu_greedy_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    """Return a unique, deterministic approximate assignment entirely on-device.

    The exact SciPy matcher copies one cost matrix per drawing from CUDA to
    CPU, which can idle a fast GPU for most of a training step.  This matcher
    repeatedly selects the currently cheapest target/query pair while keeping
    the tensors on their existing device.  It is intentionally used for the
    training loss only; validation and evaluation continue to use exact
    Hungarian matching by default so reported metrics stay comparable.
    """
    target_count, query_count = cost.shape
    if target_count == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    if target_count > query_count:
        raise ValueError("More targets than queries. Increase --num-queries or lower --max-targets.")

    with torch.no_grad():
        remaining_targets = torch.arange(target_count, dtype=torch.long, device=cost.device)
        available_queries = torch.ones(query_count, dtype=torch.bool, device=cost.device)
        target_indices: list[Tensor] = []
        query_indices: list[Tensor] = []
        for _ in range(target_count):
            available_cost = cost.detach().index_select(0, remaining_targets).masked_fill(
                ~available_queries.unsqueeze(0), float("inf")
            )
            row_cost, row_queries = available_cost.min(dim=1)
            selected_row = row_cost.argmin()
            selected_target = remaining_targets[selected_row]
            selected_query = row_queries[selected_row]
            target_indices.append(selected_target)
            query_indices.append(selected_query)
            available_queries[selected_query] = False
            # Boolean indexing keeps the selected target value on the device;
            # using it as a Python slice index would implicitly synchronize CUDA.
            remaining_targets = remaining_targets[remaining_targets != selected_target]
    return torch.stack(target_indices), torch.stack(query_indices)


def hungarian_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    """Exact minimum-cost assignment for a (targets x queries) matrix.

    This implementation avoids a SciPy runtime dependency.  Its input must
    have no more target rows than prediction columns, as in DETR matching.
    """
    target_count, query_count = cost.shape
    if target_count == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    if target_count > query_count:
        raise ValueError("More targets than queries. Increase --num-queries or lower --max-targets.")
    if linear_sum_assignment is not None:
        target_indices, query_indices = linear_sum_assignment(cost.detach().to(device="cpu", dtype=torch.float64).numpy())
        return torch.from_numpy(target_indices).long(), torch.from_numpy(query_indices).long()
    matrix = cost.detach().to(device="cpu", dtype=torch.float64).tolist()
    u = [0.0] * (target_count + 1)
    v = [0.0] * (query_count + 1)
    matching = [0] * (query_count + 1)
    way = [0] * (query_count + 1)
    for target in range(1, target_count + 1):
        matching[0] = target
        column0 = 0
        min_value = [float("inf")] * (query_count + 1)
        used = [False] * (query_count + 1)
        while True:
            used[column0] = True
            target0 = matching[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, query_count + 1):
                if used[column]:
                    continue
                candidate = matrix[target0 - 1][column - 1] - u[target0] - v[column]
                if candidate < min_value[column]:
                    min_value[column] = candidate
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(query_count + 1):
                if used[column]:
                    u[matching[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if matching[column0] == 0:
                break
        while True:
            previous = way[column0]
            matching[column0] = matching[previous]
            column0 = previous
            if column0 == 0:
                break
    target_to_query = [0] * target_count
    for query in range(1, query_count + 1):
        if matching[query]:
            target_to_query[matching[query] - 1] = query - 1
    return torch.arange(target_count, dtype=torch.long), torch.tensor(target_to_query, dtype=torch.long)


def matching_cost(class_logits: Tensor, geometry: Tensor, target: dict[str, Tensor], geometry_weight: float) -> Tensor:
    """Return a targets-by-queries cost matrix for one image."""
    target_classes = target["classes"].to(class_logits.device)
    if not target_classes.numel():
        return class_logits.new_empty((0, class_logits.shape[0]))
    probabilities = class_logits.softmax(dim=-1)
    class_cost = -probabilities[:, target_classes].transpose(0, 1)
    target_geometry = target["geometry"].to(geometry.device)
    target_mask = target["geometry_mask"].to(geometry.device)
    difference = (geometry.unsqueeze(0) - target_geometry.unsqueeze(1)).abs() * target_mask.unsqueeze(1)
    geometry_cost = difference.sum(dim=-1) / target_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return class_cost + geometry_weight * geometry_cost


class SetCriterion(nn.Module):
    def __init__(
        self, no_object_weight: float = 0.1, geometry_weight: float = 5.0, matcher: str = "exact"
    ) -> None:
        super().__init__()
        if matcher not in {"exact", "gpu-greedy"}:
            raise ValueError(f"Unknown matcher '{matcher}'. Use exact or gpu-greedy.")
        self.geometry_weight = geometry_weight
        self.matcher = matcher
        self.register_buffer("class_weight", torch.tensor([no_object_weight, 1.0, 1.0, 1.0]))

    def match(self, class_logits: Tensor, geometry: Tensor, target: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        cost = matching_cost(class_logits, geometry, target, self.geometry_weight)
        if self.matcher == "gpu-greedy":
            target_indices, query_indices = gpu_greedy_assignment(cost)
        else:
            target_indices, query_indices = hungarian_assignment(cost)
        return target_indices.to(class_logits.device), query_indices.to(class_logits.device)

    def forward(
        self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]], return_matches: bool = False
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], list[tuple[Tensor, Tensor]]]:
        class_logits = outputs["class_logits"]
        geometry = outputs["geometry"]
        class_losses: list[Tensor] = []
        geometry_numerator = geometry.new_zeros(())
        geometry_denominator = geometry.new_zeros(())
        matches: list[tuple[Tensor, Tensor]] = []
        for logits_per_image, geometry_per_image, target in zip(class_logits, geometry, targets):
            target_indices, query_indices = self.match(logits_per_image, geometry_per_image, target)
            matches.append((target_indices, query_indices))
            query_classes = torch.full(
                (logits_per_image.shape[0],), NO_OBJECT, dtype=torch.long, device=logits_per_image.device
            )
            if target_indices.numel():
                target_classes = target["classes"].to(logits_per_image.device)
                query_classes[query_indices] = target_classes[target_indices]
                target_geometry = target["geometry"].to(geometry_per_image.device)[target_indices]
                target_mask = target["geometry_mask"].to(geometry_per_image.device)[target_indices]
                geometry_numerator += ((geometry_per_image[query_indices] - target_geometry).abs() * target_mask).sum()
                geometry_denominator += target_mask.sum()
            class_losses.append(F.cross_entropy(logits_per_image, query_classes, weight=self.class_weight))
        classification_loss = torch.stack(class_losses).mean()
        geometry_loss = geometry_numerator / geometry_denominator.clamp_min(1.0)
        losses = {
            "classification": classification_loss,
            "geometry": geometry_loss,
            "total": classification_loss + self.geometry_weight * geometry_loss,
        }
        return (losses, matches) if return_matches else losses


def evaluation_metrics(
    outputs: dict[str, Tensor], targets: list[dict[str, Tensor]], criterion: SetCriterion, score_threshold: float
) -> dict[str, float]:
    """Compute a strict primitive F1 and matched normalized geometry MAE."""
    true_positive = false_positive = false_negative = 0
    geometry_error: list[float] = []
    for logits, geometry, target in zip(outputs["class_logits"], outputs["geometry"], targets):
        target_indices, query_indices = criterion.match(logits, geometry, target)
        scores, predicted_classes = logits.softmax(dim=-1).max(dim=-1)
        positive = (predicted_classes != NO_OBJECT) & (scores >= score_threshold)
        matched_true_positive = 0
        if target_indices.numel():
            target_classes = target["classes"].to(logits.device)[target_indices]
            target_geometry = target["geometry"].to(logits.device)[target_indices]
            target_mask = target["geometry_mask"].to(logits.device)[target_indices]
            pair_mae = ((geometry[query_indices] - target_geometry).abs() * target_mask).sum(dim=-1)
            pair_mae = pair_mae / target_mask.sum(dim=-1).clamp_min(1.0)
            geometry_error.extend(float(value) for value in pair_mae.detach().cpu())
            correct = (predicted_classes[query_indices] == target_classes) & (scores[query_indices] >= score_threshold)
            # 0.03 is roughly 15 px at a 512 px model input.
            correct &= pair_mae <= 0.03
            matched_true_positive = int(correct.sum().item())
            false_negative += int(target_classes.numel()) - matched_true_positive
        true_positive += matched_true_positive
        false_positive += int(positive.sum().item()) - matched_true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "matched_geometry_mae": float(np.mean(geometry_error)) if geometry_error else float("nan"),
    }


def train_epoch(
    model: PrimitiveDETR,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: SetCriterion,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    progress: RichProgress | None = None,
    task_id: int | None = None,
) -> dict[str, float]:
    model.train()
    totals = {"classification": 0.0, "geometry": 0.0, "total": 0.0}
    batches = 0
    for batch_number, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        autocast = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else nullcontext()
        with autocast:
            outputs = model(images)
            losses = criterion(outputs, targets)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizer)
        scaler.update()
        for name in totals:
            totals[name] += float(losses[name].detach())
        batches += 1
        if progress is not None and task_id is not None:
            progress.advance(task_id, images.shape[0])
            if batch_number % 25 == 0:
                progress.update(task_id, description=f"Training: loss {totals['total'] / batches:.4f}")
        elif batch_number % 100 == 0:
            print(f"  batch {batch_number:,}: loss={totals['total'] / batches:.4f}", flush=True)
    return {name: value / max(batches, 1) for name, value in totals.items()}


@torch.inference_mode()
def validate_epoch(
    model: PrimitiveDETR,
    loader: DataLoader[Any],
    criterion: SetCriterion,
    device: torch.device,
    score_threshold: float,
    progress: RichProgress | None = None,
    task_id: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {"classification": 0.0, "geometry": 0.0, "total": 0.0}
    metrics = {"true_positive": 0.0, "false_positive": 0.0, "false_negative": 0.0, "mae_sum": 0.0, "mae_count": 0.0}
    batches = 0
    owns_progress = progress is None
    progress_context = RichProgress() if owns_progress else nullcontext(progress)
    with progress_context as active_progress:
        active_task = task_id if task_id is not None else active_progress.add_task("Evaluating", total=len(loader.dataset))
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            losses, matches = criterion(outputs, targets, return_matches=True)
            for name in totals:
                totals[name] += float(losses[name])
            # Reuse the assignments calculated for the loss. This avoids a
            # second matching pass for every drawing in the validation batch.
            for logits, geometry, target, (target_indices, query_indices) in zip(
                outputs["class_logits"], outputs["geometry"], targets, matches
            ):
                scores, predicted_classes = logits.softmax(dim=-1).max(dim=-1)
                positive = (predicted_classes != NO_OBJECT) & (scores >= score_threshold)
                matched_true_positive = 0
                if target_indices.numel():
                    target_classes = target["classes"].to(device)[target_indices]
                    target_geometry = target["geometry"].to(device)[target_indices]
                    target_mask = target["geometry_mask"].to(device)[target_indices]
                    pair_mae = ((geometry[query_indices] - target_geometry).abs() * target_mask).sum(dim=-1)
                    pair_mae = pair_mae / target_mask.sum(dim=-1).clamp_min(1.0)
                    correct = (predicted_classes[query_indices] == target_classes) & (scores[query_indices] >= score_threshold)
                    correct &= pair_mae <= 0.03
                    matched_true_positive = int(correct.sum().item())
                    metrics["false_negative"] += int(target_classes.numel()) - matched_true_positive
                    metrics["mae_sum"] += float(pair_mae.sum())
                    metrics["mae_count"] += float(pair_mae.numel())
                metrics["true_positive"] += matched_true_positive
                metrics["false_positive"] += int(positive.sum().item()) - matched_true_positive
            batches += 1
            if active_task is not None:
                active_progress.advance(active_task, images.shape[0])
            elif batches % 250 == 0:
                print(f"[EVALUATE] {min(batches * loader.batch_size, len(loader.dataset)):,}/{len(loader.dataset):,} samples", flush=True)
    precision = metrics["true_positive"] / max(metrics["true_positive"] + metrics["false_positive"], 1.0)
    recall = metrics["true_positive"] / max(metrics["true_positive"] + metrics["false_negative"], 1.0)
    return {
        **{f"val_{name}": value / max(batches, 1) for name, value in totals.items()},
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "matched_geometry_mae": metrics["mae_sum"] / max(metrics["mae_count"], 1.0),
    }


def checkpoint_payload(
    model: PrimitiveDETR,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    model_args: argparse.Namespace,
    metrics: dict[str, float],
    best_f1: float,
) -> dict[str, Any]:
    return {
        "format": "paracad_primitive_detr_v1",
        "epoch": epoch,
        "model": {
            "d_model": model.d_model,
            "num_queries": model.num_queries,
            "decoder_layers": model_args.decoder_layers,
            "image_size": model_args.image_size,
        },
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "criterion": {
            "geometry_weight": model_args.geometry_weight,
            "no_object_weight": model_args.no_object_weight,
            "training_matcher": model_args.matcher,
            "validation_matcher": model_args.validation_matcher,
        },
        "metrics": metrics,
        "best_f1": best_f1,
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> PrimitiveDETR:
    config = checkpoint["model"]
    model = PrimitiveDETR(
        d_model=int(config["d_model"]),
        num_queries=int(config["num_queries"]),
        decoder_layers=int(config["decoder_layers"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def train(args: argparse.Namespace) -> int:
    if not args.dataset.is_dir():
        raise ValueError(f"Dataset directory not found: {args.dataset}")
    if args.max_targets > args.num_queries:
        raise ValueError("--max-targets cannot exceed --num-queries.")
    if args.init_checkpoint is not None and args.resume is not None:
        raise ValueError("Use either --init-checkpoint or --resume, not both.")
    seed_everything(args.seed)
    device = choose_device(args.device)
    configure_acceleration(device)
    print(f"[INFO] Using {device}; loading split indices...", flush=True)
    train_dataset = ParaCADDrawingDataset(
        args.dataset, "train", args.image_size, args.max_targets, args.train_max_samples, args.seed
    )
    val_dataset = ParaCADDrawingDataset(args.dataset, "val", args.image_size, args.max_targets, args.val_max_samples, args.seed + 1)
    print(
        f"[INFO] train={len(train_dataset):,}, val={len(val_dataset):,}; "
        f"workers={args.num_workers}; training matcher={args.matcher}; validation matcher={args.validation_matcher}",
        flush=True,
    )
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device)
    model = PrimitiveDETR(args.d_model, args.num_queries, args.decoder_layers).to(device)
    checkpoint_path = args.resume or args.init_checkpoint
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_model = checkpoint.get("model", {})
        expected = {"d_model": args.d_model, "num_queries": args.num_queries, "decoder_layers": args.decoder_layers}
        mismatched = {key: (checkpoint_model.get(key), value) for key, value in expected.items() if int(checkpoint_model.get(key, -1)) != value}
        if mismatched:
            option = "--resume" if args.resume is not None else "--init-checkpoint"
            raise ValueError(f"{option} architecture differs from current arguments: {mismatched}")
        model.load_state_dict(checkpoint["state_dict"])
        if args.init_checkpoint is not None:
            print(f"[INFO] Initialized model weights from {args.init_checkpoint}; optimizer state is intentionally reset.", flush=True)
    criterion = SetCriterion(args.no_object_weight, args.geometry_weight, args.matcher).to(device)
    validation_criterion = SetCriterion(
        args.no_object_weight, args.geometry_weight, args.validation_matcher
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and not args.no_amp)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    start_epoch = 1
    if args.resume is not None:
        if checkpoint is None or "optimizer" not in checkpoint:
            raise ValueError("--resume requires a training checkpoint containing optimizer state.")
        completed_epoch = int(checkpoint.get("epoch", 0))
        if completed_epoch < 1:
            raise ValueError("--resume checkpoint does not record a completed epoch.")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            # Backward-compatible continuation of checkpoints written before
            # scheduler state was saved. The next scheduler.step() advances
            # from the completed epoch rather than restarting the LR curve.
            scheduler.last_epoch = completed_epoch
            scheduler._step_count = completed_epoch + 1
            scheduler._last_lr = scheduler._get_closed_form_lr()
            for group, learning_rate in zip(optimizer.param_groups, scheduler._last_lr):
                group["lr"] = learning_rate
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        best_f1 = float(checkpoint.get("best_f1", checkpoint.get("metrics", {}).get("f1", -1.0)))
        start_epoch = completed_epoch + 1
        print(
            f"[INFO] Resuming {args.resume} after epoch {completed_epoch}; "
            f"continuing at epoch {start_epoch}/{args.epochs} (best F1={best_f1:.4f}).",
            flush=True,
        )
        if start_epoch > args.epochs:
            print("[INFO] Resume checkpoint already reached --epochs; no training is needed.", flush=True)
            return 0
    with RichProgress(enabled=not args.no_progress) as progress:
        epoch_task = progress.add_task("Training epochs", total=args.epochs)
        if start_epoch > 1:
            progress.advance(epoch_task, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"[EPOCH {epoch}/{args.epochs}]", flush=True)
            train_task = progress.add_task(f"Training epoch {epoch}/{args.epochs}", total=len(train_dataset))
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler, not args.no_amp, progress, train_task)
            progress.complete(train_task, f"Training epoch {epoch}/{args.epochs}: loss {train_metrics['total']:.4f}")
            validation_task = progress.add_task(f"Validation epoch {epoch}/{args.epochs}", total=len(val_dataset))
            val_metrics = validate_epoch(
                model, val_loader, validation_criterion, device, args.score_threshold, progress, validation_task
            )
            progress.complete(validation_task, f"Validation epoch {epoch}/{args.epochs}: F1 {val_metrics['f1']:.4f}")
            scheduler.step()
            metrics = {**{f"train_{name}": value for name, value in train_metrics.items()}, **val_metrics}
            print(json.dumps(metrics, allow_nan=False, sort_keys=True), flush=True)
            last_path = args.checkpoint_dir / "last.pt"
            is_best = metrics["f1"] > best_f1
            best_f1 = max(best_f1, metrics["f1"])
            payload = checkpoint_payload(model, optimizer, scheduler, scaler, epoch, args, metrics, best_f1)
            torch.save(payload, last_path)
            if is_best:
                torch.save(payload, args.checkpoint_dir / "best.pt")
                print(f"[INFO] Saved new best checkpoint (F1={best_f1:.4f}).", flush=True)
            progress.advance(epoch_task)
    return 0


def decode_primitives(
    outputs: dict[str, Tensor], score_threshold: float, transform: dict[str, float]
) -> list[dict[str, Any]]:
    logits = outputs["class_logits"][0]
    geometry = outputs["geometry"][0]
    scores, classes = logits.softmax(dim=-1).max(dim=-1)
    result: list[dict[str, Any]] = []
    for index, (score, primitive_class, vector) in enumerate(zip(scores, classes, geometry)):
        score_value = float(score)
        class_value = int(primitive_class)
        if class_value == NO_OBJECT or score_value < score_threshold:
            continue
        values = [float(value) for value in vector]
        item: dict[str, Any] = {"query_index": index, "type": CLASS_NAMES[class_value], "score": score_value}
        if class_value == LINE:
            item["geometry"] = {
                "x1": (values[0] - transform["x_offset"]) / transform["x_scale"],
                "y1": (values[1] - transform["y_offset"]) / transform["y_scale"],
                "x2": (values[2] - transform["x_offset"]) / transform["x_scale"],
                "y2": (values[3] - transform["y_offset"]) / transform["y_scale"],
            }
        elif class_value == CIRCLE:
            item["geometry"] = {
                "cx": (values[4] - transform["x_offset"]) / transform["x_scale"],
                "cy": (values[5] - transform["y_offset"]) / transform["y_scale"],
                "radius": values[6] / transform["radius_scale"],
            }
        else:
            def angle(sine: float, cosine: float) -> float:
                return math.degrees(math.atan2(2.0 * sine - 1.0, 2.0 * cosine - 1.0)) % 360.0

            item["geometry"] = {
                "cx": (values[4] - transform["x_offset"]) / transform["x_scale"],
                "cy": (values[5] - transform["y_offset"]) / transform["y_scale"],
                "radius": values[6] / transform["radius_scale"],
                "start_param": angle(values[7], values[8]),
                "end_param": angle(values[9], values[10]),
            }
        result.append(item)
    return sorted(result, key=lambda item: item["score"], reverse=True)


@torch.no_grad()
def predict(args: argparse.Namespace) -> int:
    if not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint not found: {args.checkpoint}")
    if not args.image.is_file():
        raise ValueError(f"Image not found: {args.image}")
    with RichProgress(enabled=not args.no_progress) as progress:
        task = progress.add_task("Predicting drawing primitives", total=4)
        device = choose_device(args.device)
        configure_acceleration(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = build_model_from_checkpoint(checkpoint, device)
        progress.advance(task)
        image_size = int(checkpoint["model"]["image_size"])
        with Image.open(args.image) as image:
            tensor, transform = image_to_tensor(image, image_size)
            tensor = tensor.unsqueeze(0).to(device)
        progress.advance(task)
        primitives = decode_primitives(model(tensor), args.score_threshold, transform)
        progress.advance(task)
        payload = {
            "model": checkpoint.get("format", "unknown"),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "coordinate_space": "normalized",
            "image": str(args.image),
            "primitives": primitives,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        progress.advance(task)
        progress.complete(task, "Primitive prediction complete")
    print(f"[DONE] Wrote {len(primitives):,} primitives to {args.output}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    """Measure a saved checkpoint against ParaCAD's held-out test split."""
    if not args.dataset.is_dir():
        raise ValueError(f"Dataset directory not found: {args.dataset}")
    if not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint not found: {args.checkpoint}")
    device = choose_device(args.device)
    configure_acceleration(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device)
    model_config = checkpoint["model"]
    model = maybe_compile_for_evaluation(model, int(model_config["image_size"]), device, args.compile)
    criterion_config = checkpoint.get("criterion", {})
    criterion = SetCriterion(
        no_object_weight=float(criterion_config.get("no_object_weight", 0.1)),
        geometry_weight=float(criterion_config.get("geometry_weight", 5.0)),
        matcher=args.matcher,
    ).to(device)
    dataset = ParaCADDrawingDataset(
        args.dataset,
        "test",
        int(model_config["image_size"]),
        args.max_targets,
        args.max_samples,
        args.seed,
    )
    loader = make_loader(dataset, args.batch_size, False, args.num_workers, device)
    with RichProgress(enabled=not args.no_progress) as progress:
        task = progress.add_task("Evaluating test drawings", total=len(dataset))
        metrics = validate_epoch(model, loader, criterion, device, args.score_threshold, progress, task)
        progress.complete(task, "Test evaluation complete")
    print(json.dumps({"checkpoint": str(args.checkpoint), "test_samples": len(dataset), **metrics}, allow_nan=False, sort_keys=True))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or run a DETR-style image-to-CAD-primitive model.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train with ParaCAD's Zarr images and primitive labels.")
    train_parser.add_argument("--dataset", type=Path, default=Path("ParaCAD_full_v3.zarr"))
    train_parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    checkpoint_group = train_parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Optional compatible checkpoint whose model weights initialize a new run; optimizer state is reset.",
    )
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        help="Resume a completed epoch from last.pt, restoring model, optimizer, learning-rate schedule, and AMP scaler.",
    )
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS, help="Parallel Zarr/Pillow loader workers.")
    train_parser.add_argument(
        "--matcher",
        choices=("gpu-greedy", "exact"),
        default="gpu-greedy",
        help="Training assignment matcher. gpu-greedy keeps matching on CUDA; exact is slower but optimal.",
    )
    train_parser.add_argument(
        "--validation-matcher",
        choices=("exact", "gpu-greedy"),
        default="exact",
        help="Validation assignment matcher. exact preserves comparable F1/MAE metrics.",
    )
    train_parser.add_argument("--image-size", type=int, default=512)
    train_parser.add_argument("--num-queries", type=int, default=128)
    train_parser.add_argument("--max-targets", type=int, default=96, help="Drawings with more primitives are truncated for this first-stage model.")
    train_parser.add_argument("--d-model", type=int, default=256)
    train_parser.add_argument("--decoder-layers", type=int, default=6)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--geometry-weight", type=float, default=5.0)
    train_parser.add_argument("--no-object-weight", type=float, default=0.1)
    train_parser.add_argument("--score-threshold", type=float, default=0.5)
    train_parser.add_argument("--train-max-samples", type=int, default=100_000, help="0 uses every training record.")
    train_parser.add_argument("--val-max-samples", type=int, default=10_000, help="0 uses every validation record.")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    train_parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    add_progress_argument(train_parser)
    predict_parser = commands.add_parser("predict", help="Predict primitives for a standalone drawing image.")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, default=Path("predicted_primitives.json"))
    predict_parser.add_argument("--score-threshold", type=float, default=0.5)
    predict_parser.add_argument("--device", default="auto")
    add_progress_argument(predict_parser)
    evaluate_parser = commands.add_parser("evaluate", help="Measure a checkpoint against the held-out ParaCAD test split.")
    evaluate_parser.add_argument("--dataset", type=Path, default=Path("ParaCAD_full_v3.zarr"))
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=32)
    evaluate_parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS, help="Parallel Zarr/Pillow loader workers.")
    evaluate_parser.add_argument(
        "--matcher",
        choices=("exact", "gpu-greedy"),
        default="exact",
        help="Assignment matcher for loss and metrics. exact is recommended for reported evaluation metrics.",
    )
    evaluate_parser.add_argument("--max-targets", type=int, default=96)
    evaluate_parser.add_argument("--max-samples", type=int, default=0, help="0 uses every test record.")
    evaluate_parser.add_argument("--score-threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--seed", type=int, default=44)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--compile", action="store_true", help="Use torch.compile; startup is slower but long evaluations are faster.")
    add_progress_argument(evaluate_parser)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        if args.command == "train":
            return train(args)
        if args.command == "predict":
            return predict(args)
        return evaluate(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
