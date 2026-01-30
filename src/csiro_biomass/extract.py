"""Explicit command entry point for identified frozen image features."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .data import project_output, read_image_paths, save_feature_cache


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="train.csv or test.csv with image_path")
    parser.add_argument("--data-root", type=Path, required=True, help="Read-only root containing the image paths")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local Hugging Face directory or Keras weights file")
    parser.add_argument("--model-id", required=True, help="User-supplied identifier for the supplied checkpoint")
    parser.add_argument("--output", type=Path, default=Path("features/extracted"))
    parser.add_argument("--method", choices=["dinov2-whole", "dinov2-tiled-mean", "dinov2-tiled-advanced", "convnext"], required=True)
    parser.add_argument("--batch-size", type=int, default=4, help="Original images per batch, including all their tiles")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda; no automatic GPU selection")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("ROWS", "COLS"), help="Required explicit layout for tiled DINOv2")
    parser.add_argument("--convnext-variant", choices=["large", "xlarge"])
    parser.add_argument("--pixel-scale", choices=["raw", "divide_255"])
    parser.add_argument("--include-preprocessing", choices=["true", "false"])
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--stage-layers", nargs=2, metavar=("STAGE2", "STAGE3"))
    parser.add_argument("--zoom", type=float, help="ConvNeXt central crop factor from the zoom variants")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.height < 1 or args.width < 1:
        parser.error("Batch size and image dimensions must be positive.")
    output_dir = project_output(args.output)
    project_root = Path(__file__).resolve().parents[2]
    cache_root = project_root / ".cache"
    for cache_path in [
        cache_root, cache_root / "huggingface", cache_root / "huggingface" / "hub",
        cache_root / "huggingface" / "transformers", cache_root / "torch",
        cache_root / "keras", cache_root / "tmp",
    ]:
        project_output(cache_path)
    # Keep libraries' incidental caches local, even when existing global paths are set.
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "huggingface" / "transformers")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["KERAS_HOME"] = str(cache_root / "keras")
    for variable in ["TMP", "TEMP", "TMPDIR"]:
        os.environ[variable] = str(cache_root / "tmp")
    (cache_root / "tmp").mkdir(parents=True, exist_ok=True)
    image_paths = read_image_paths(args.metadata)
    metadata = {
        "method": args.method, "model_identifier": args.model_id,
        "checkpoint_name": args.checkpoint.name, "frozen_inference": True,
        "batch_size_original_images": args.batch_size,
    }
    if args.method.startswith("dinov2"):
        from .dinov2 import extract_tiled, extract_whole_image, load_local_dinov2

        if args.zoom is not None or args.stage_layers is not None:
            parser.error("Zoom and stage-layer arguments apply only to ConvNeXt.")
        if args.method == "dinov2-whole" and args.grid is not None:
            parser.error("Whole-image DINOv2 does not use a tile grid.")
        if args.method != "dinov2-whole" and (args.grid is None or min(args.grid) < 1):
            parser.error("Tiled DINOv2 requires --grid ROWS COLS with positive integers.")
        processor, model = load_local_dinov2(args.checkpoint, args.device)
        if args.method == "dinov2-whole":
            features, blocks = extract_whole_image(image_paths, args.data_root, processor, model, args.batch_size)
        else:
            pooling = "advanced" if args.method.endswith("advanced") else "mean"
            features, blocks = extract_tiled(
                image_paths, args.data_root, processor, model, *args.grid, pooling, args.batch_size
            )
        metadata.update({
            "backend": "pytorch_transformers", "device": str(model.device),
            "model_type": model.config.model_type, "hidden_size": model.config.hidden_size,
            "patch_size": model.config.patch_size,
            "preprocessing": {"do_resize": False, "do_center_crop": False,
                              "remaining_processor_settings": processor.to_dict()},
            "tile_grid": args.grid,
            "tile_order": "row-major within each original image" if args.grid else None,
            "tile_crop": "central 1000x2000 crop divisible by grid and patch size" if args.grid else None,
            "provenance": "global patch mean and mean tile pooler output" if args.method.endswith("tiled-mean") else "report DINOv2 feature definitions",
        })
    else:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        from .convnext import build_local_convnext, extract_convnext

        if args.grid is not None:
            parser.error("ConvNeXt does not use the DINOv2 tile grid.")
        if args.convnext_variant is None or args.pixel_scale is None or args.include_preprocessing is None:
            parser.error("ConvNeXt requires --convnext-variant, --pixel-scale and --include-preprocessing.")
        model, blocks = build_local_convnext(
            args.checkpoint, args.convnext_variant, args.height, args.width,
            args.include_preprocessing == "true", args.stage_layers,
        )
        features = extract_convnext(
            image_paths, args.data_root, model, args.height, args.width,
            args.pixel_scale, args.batch_size, args.zoom,
        )
        metadata.update({
            "backend": "keras_tensorflow", "variant": args.convnext_variant,
            "preprocessing": {"input_height": args.height, "input_width": args.width,
                              "pixel_scale": args.pixel_scale,
                              "include_preprocessing": args.include_preprocessing == "true",
                              "initial_resize": "nearest", "zoom_resize": "TensorFlow bilinear" if args.zoom else None},
            "zoom": args.zoom, "stage_layers": args.stage_layers,
            "provenance": "Keras ConvNeXt with explicit preprocessing profiles",
        })
    save_feature_cache(output_dir, image_paths, features, {**metadata, "blocks": blocks})
    print(f"Wrote {len(image_paths)} identified feature rows to {output_dir}")


if __name__ == "__main__":
    main()
