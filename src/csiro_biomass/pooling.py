"""DINOv2 token pooling."""

from __future__ import annotations

from typing import Any, Sequence


def block_schema(names: Sequence[str], widths: Sequence[int]) -> list[dict[str, Any]]:
    """Record the ordered feature blocks using actual tensor dimensions."""
    offset = 0
    blocks = []
    for name, width in zip(names, widths):
        blocks.append({"name": name, "width": int(width), "start": offset, "end": offset + int(width)})
        offset += int(width)
    return blocks


def pool_whole_image(outputs: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Concatenate mean patch tokens, then the model's pooler_output."""
    import torch

    patch_mean = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
    pooled_cls = outputs.pooler_output
    if pooled_cls is None:
        raise ValueError("The local checkpoint must supply pooler_output.")
    features = torch.cat((patch_mean, pooled_cls), dim=1)
    schema = block_schema(["patch_mean", "pooler_output"], [patch_mean.shape[1], pooled_cls.shape[1]])
    return features, schema


def group_tile_outputs(outputs: Any, image_count: int, grid_cells: int) -> tuple[Any, Any]:
    """Flatten all tiles' patch tokens per image; average tile pooler outputs."""
    raw_tokens = outputs.last_hidden_state[:, 1:, :]
    grouped = raw_tokens.reshape(image_count, grid_cells, raw_tokens.shape[1], -1)
    global_tokens = grouped.reshape(image_count, -1, grouped.shape[-1])
    if outputs.pooler_output is None:
        raise ValueError("The local checkpoint must supply pooler_output.")
    mean_cls = outputs.pooler_output.reshape(image_count, grid_cells, -1).mean(dim=1)
    return global_tokens, mean_cls


def pool_tiled_mean(global_tokens: Any, mean_cls: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Basic tiled variant: global patch mean followed by mean CLS."""
    import torch

    patch_mean = global_tokens.mean(dim=1)
    features = torch.cat((patch_mean, mean_cls), dim=1)
    schema = block_schema(["global_patch_mean", "mean_tile_pooler_output"], [patch_mean.shape[1], mean_cls.shape[1]])
    return features, schema


def pool_tiled_advanced(global_tokens: Any, mean_cls: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Return the report's 17 blocks; tokens have shape [images, tokens, width]."""
    import torch

    mean = torch.mean(global_tokens, dim=1)
    # The original torch.std default uses a sample standard deviation.
    standard_deviation = torch.std(global_tokens, dim=1, unbiased=True)
    centered = global_tokens - mean.unsqueeze(1)
    skewness = torch.mean(centered ** 3, dim=1) / (standard_deviation ** 3 + 1e-6)
    kurtosis = torch.mean(centered ** 4, dim=1) / (standard_deviation ** 4 + 1e-6)
    gem = global_tokens.clamp(min=1e-6).pow(3).mean(dim=1).pow(1.0 / 3)
    minimum = torch.min(global_tokens, dim=1).values
    maximum = torch.max(global_tokens, dim=1).values
    quantile_points = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=global_tokens.device)
    quantiles = torch.quantile(global_tokens, quantile_points, dim=1).permute(1, 0, 2)
    positive_proportion = (global_tokens > 0).float().mean(dim=1)
    l2 = torch.norm(global_tokens, p=2, dim=1)
    l1 = torch.norm(global_tokens, p=1, dim=1)
    ratio = l1 / (l2 + 1e-6)
    tensors = [
        mean, standard_deviation, skewness, kurtosis, gem, minimum, maximum,
        *[quantiles[:, index, :] for index in range(5)],
        positive_proportion, l2, l1, ratio, mean_cls,
    ]
    names = [
        "global_patch_mean", "sample_std", "skewness", "non_excess_kurtosis", "gem_p3",
        "minimum", "maximum", "quantile_0.10", "quantile_0.25", "quantile_0.50",
        "quantile_0.75", "quantile_0.90", "positive_proportion", "l2_norm", "l1_norm",
        "l1_l2_ratio", "mean_tile_pooler_output",
    ]
    features = torch.cat(tensors, dim=1)
    return features, block_schema(names, [tensor.shape[1] for tensor in tensors])
