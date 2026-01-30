"""Frozen ConvNeXt image features with local Keras weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .data import resolve_image
from .pooling import block_schema


def build_local_convnext(
    checkpoint: Path, variant: str, height: int, width: int,
    include_preprocessing: bool, stage_layers: Sequence[str] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build the selected Keras backbone and load local, include_top=False weights."""
    import keras

    if not checkpoint.is_file():
        raise FileNotFoundError("ConvNeXt requires a local Keras backbone weights file.")
    constructors = {"large": keras.applications.ConvNeXtLarge, "xlarge": keras.applications.ConvNeXtXLarge}
    if variant not in constructors:
        raise ValueError("Select the Large or XLarge architecture explicitly.")
    backbone = constructors[variant](
        weights=None, include_top=False, input_shape=(height, width, 3),
        include_preprocessing=include_preprocessing,
    )
    backbone.load_weights(checkpoint)
    backbone.trainable = False
    if stage_layers is None:
        feature_tensor = keras.layers.GlobalAveragePooling2D()(backbone.output)
        blocks = block_schema(["global_mean"], [int(feature_tensor.shape[-1])])
    else:
        if len(stage_layers) != 2:
            raise ValueError("Stage pooling requires two explicit, existing Keras layer names.")
        tensors, names = [], []
        for stage_name in stage_layers:
            stage = backbone.get_layer(stage_name).output
            tensors.extend([
                keras.layers.GlobalAveragePooling2D()(stage),
                keras.layers.GlobalMaxPooling2D()(stage),
            ])
            names.extend([f"{stage_name}_mean", f"{stage_name}_max"])
        feature_tensor = keras.layers.Concatenate()(tensors)
        blocks = block_schema(names, [int(tensor.shape[-1]) for tensor in tensors])
    feature_model = keras.Model(backbone.input, feature_tensor)
    feature_model.trainable = False
    return feature_model, blocks


def extract_convnext(
    image_paths: Sequence[str], data_root: Path, model: Any,
    height: int, width: int, pixel_scale: str, batch_size: int = 8,
    zoom: float | None = None,
) -> Any:
    """Extract batched features with optional central cropping and TensorFlow resizing."""
    import keras
    import numpy as np

    if pixel_scale not in {"raw", "divide_255"}:
        raise ValueError("Choose raw pixels or the supplied scripts' divide_255 scaling.")
    if zoom is not None and zoom < 1:
        raise ValueError("Central zoom must be at least one.")
    batches = []
    for start in range(0, len(image_paths), batch_size):
        images = []
        for image_path in image_paths[start:start + batch_size]:
            initial_size = (1000, 2000) if zoom is not None else (height, width)
            image = keras.utils.load_img(
                resolve_image(data_root, image_path), color_mode="rgb",
                target_size=initial_size, interpolation="nearest",
            )
            pixels = keras.utils.img_to_array(image)
            if zoom is not None:
                import tensorflow as tf

                crop_height, crop_width = round(1000 / zoom), round(2000 / zoom)
                if crop_height < 1 or crop_width < 1:
                    raise ValueError("The requested zoom leaves no image region.")
                top, left = (1000 - crop_height) // 2, (2000 - crop_width) // 2
                pixels = pixels[top:top + crop_height, left:left + crop_width, :]
                pixels = tf.image.resize(pixels, (height, width)).numpy()
            if pixel_scale == "divide_255":
                pixels = pixels / 255
            images.append(pixels)
        # The model output sets the width; no historical 1536/4608 allocation is assumed.
        batches.append(np.asarray(model(np.stack(images), training=False)))
    return np.vstack(batches)
