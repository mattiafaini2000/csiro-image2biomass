"""Image identities and the CSV/NumPy interchange with R."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence


def read_image_paths(metadata_file: Path) -> list[str]:
    """Return unique image_path values in their first-occurrence CSV order."""
    with metadata_file.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError("Metadata must contain an image_path column.")
        image_paths = list(dict.fromkeys(row["image_path"] for row in reader))
    if not image_paths or any(not image_path for image_path in image_paths):
        raise ValueError("Metadata must contain non-empty image identities.")
    return image_paths


def resolve_image(data_root: Path, image_path: str) -> Path:
    """Resolve a metadata identity below the read-only image data root."""
    candidate = (data_root / image_path).resolve()
    if not candidate.is_relative_to(data_root.resolve()):
        raise ValueError(f"Image identity leaves the data root: {image_path}")
    return candidate


def project_output(output_dir: Path) -> Path:
    """Require outputs, including symlink targets, to remain in this checkout."""
    project_root = Path(__file__).resolve().parents[2]
    destination = output_dir.resolve()
    if not destination.is_relative_to(project_root):
        raise ValueError("The output directory must be inside the project checkout.")
    return destination


def save_feature_cache(
    output_dir: Path,
    image_paths: Sequence[str],
    features: Any,
    schema: dict[str, Any],
) -> None:
    """Write identified features; CSV column order matches NPZ column order."""
    import numpy as np

    destination = project_output(output_dir)
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[0] != len(image_paths):
        raise ValueError("Feature rows must match the image manifest.")
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("Feature image identities must be unique.")
    if not np.isfinite(matrix).all():
        raise ValueError("The extracted feature matrix contains non-finite values.")
    if any((destination / name).exists() or (destination / name).is_symlink() for name in [
        "features.npz", "features.csv", "manifest.csv", "schema.json"
    ]):
        raise FileExistsError("Choose a new output directory for this feature cache.")
    destination.mkdir(parents=True, exist_ok=True)
    columns = [f"feature_{index + 1:06d}" for index in range(matrix.shape[1])]
    np.savez(destination / "features.npz", features=matrix, image_path=np.asarray(image_paths))
    with (destination / "features.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", *columns])
        for image_path, row in zip(image_paths, matrix):
            writer.writerow([image_path, *row.tolist()])
    with (destination / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_index", "image_path"])
        writer.writerows(enumerate(image_paths, start=1))
    schema = {
        **schema,
        "feature_dimension": int(matrix.shape[1]),
        "image_count": len(image_paths),
        "dtype": str(matrix.dtype),
        "feature_columns": columns,
        "identity_column": "image_path",
        "row_order": "first occurrence in supplied metadata; align by image_path",
        "block_offsets": "zero-based, end exclusive",
    }
    (destination / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
