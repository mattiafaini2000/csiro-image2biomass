"""Local-checkpoint DINOv2 extraction; no model is loaded at import time."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .data import resolve_image
from .pooling import group_tile_outputs, pool_tiled_advanced, pool_tiled_mean, pool_whole_image


def load_local_dinov2(checkpoint: Path, device: str) -> tuple[Any, Any]:
    """Load processor and frozen backbone exclusively from a local directory."""
    from transformers import AutoImageProcessor, AutoModel

    if not checkpoint.is_dir():
        raise FileNotFoundError("DINOv2 requires a local Hugging Face checkpoint directory.")
    processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(checkpoint, local_files_only=True)
    if model.config.model_type != "dinov2":
        raise ValueError("Supply the report's DINOv2 architecture, without register-token variants.")
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    return processor, model


def extract_whole_image(
    image_paths: Sequence[str], data_root: Path, processor: Any, model: Any, batch_size: int = 4
) -> tuple[Any, list[dict[str, Any]]]:
    """Extract whole-image features without resizing, cropping, or extra scaling."""
    import numpy as np
    import torch
    from PIL import Image

    model.eval()
    batches = []
    schema = []
    for start in range(0, len(image_paths), batch_size):
        images = []
        for image_path in image_paths[start:start + batch_size]:
            with Image.open(resolve_image(data_root, image_path)) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt", do_resize=False, do_center_crop=False).to(model.device)
        with torch.inference_mode():
            features, schema = pool_whole_image(model(**inputs))
        batches.append(features.cpu().numpy())
    return np.vstack(batches), schema


def center_crop_tiles(image: Any, n_rows: int, n_cols: int, patch_size: int) -> list[Any]:
    """Crop a 1000x2000 image to grid/patch multiples, then split in row-major order."""
    if image.size != (2000, 1000):
        raise ValueError("Tiled extraction expects 1000x2000 images.")
    width, height = image.size
    crop_height = (height // (n_rows * patch_size)) * n_rows * patch_size
    crop_width = (width // (n_cols * patch_size)) * n_cols * patch_size
    if crop_height == 0 or crop_width == 0:
        raise ValueError("The requested grid leaves no patch-sized image region.")
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    image = image.crop((left, top, left + crop_width, top + crop_height))
    tile_height, tile_width = crop_height // n_rows, crop_width // n_cols
    return [
        image.crop((col * tile_width, row * tile_height, (col + 1) * tile_width, (row + 1) * tile_height))
        for row in range(n_rows) for col in range(n_cols)
    ]


def extract_tiled(
    image_paths: Sequence[str], data_root: Path, processor: Any, model: Any,
    n_rows: int, n_cols: int, pooling: str, batch_size: int = 4,
) -> tuple[Any, list[dict[str, Any]]]:
    """Group every tile by original image before global token pooling."""
    import numpy as np
    import torch
    from PIL import Image

    if n_rows < 1 or n_cols < 1 or pooling not in {"mean", "advanced"}:
        raise ValueError("Supply a positive row/column grid and mean or advanced pooling.")
    patch_size = int(model.config.patch_size)
    if patch_size != 14:
        raise ValueError("The crop rule requires a patch-size-14 DINOv2 checkpoint.")
    model.eval()
    batches = []
    schema = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tiles = []
        for image_path in batch_paths:
            with Image.open(resolve_image(data_root, image_path)) as image:
                tiles.extend(center_crop_tiles(image.convert("RGB"), n_rows, n_cols, patch_size))
        inputs = processor(images=tiles, return_tensors="pt", do_resize=False, do_center_crop=False).to(model.device)
        with torch.inference_mode():
            global_tokens, mean_cls = group_tile_outputs(model(**inputs), len(batch_paths), n_rows * n_cols)
            pool = pool_tiled_advanced if pooling == "advanced" else pool_tiled_mean
            features, schema = pool(global_tokens, mean_cls)
        batches.append(features.cpu().numpy())
    return np.vstack(batches), schema
