from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_from_disk

from vlm_ft.data.schema import DatasetInfo, Sample, resolve_image_path, validate_sample

logger = logging.getLogger(__name__)

PROCESSED_VARIANT = "clinical_context"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def _rewrite_image_paths(sample: dict[str, Any], image_root: Path | None) -> dict[str, Any]:
    if image_root is None:
        return sample
    messages = []
    for message in sample["messages"]:
        content = message["content"]
        if isinstance(content, str):
            messages.append(message)
            continue
        new_content = []
        for part in content:
            if part.get("type") == "image" and part.get("image"):
                resolved = resolve_image_path(part["image"], image_root)
                new_content.append({**part, "image": str(resolved)})
            else:
                new_content.append(part)
        messages.append({**message, "content": new_content})
    return {**sample, "messages": messages}


def load_dataset_info(path: Path) -> DatasetInfo | None:
    info_path = path / "dataset_info.json"
    if not info_path.exists():
        return None
    with info_path.open("r", encoding="utf-8") as f:
        return DatasetInfo.model_validate(json.load(f))


def resolve_processed_root(root: str | Path) -> Path:
    """Resolve ``data`` or ``data/processed`` to the processed datasets directory."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    if root.name == "processed":
        return root
    candidate = root / "processed"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Could not find processed datasets under {root}. "
        "Expected a `processed/` directory (e.g. data/processed)."
    )


def discover_context_datasets(
    root: str | Path,
    datasets: list[str] | None = None,
) -> list[Path]:
    """Find every ``{dataset}/clinical_context/`` folder under the processed root.

    Expected layout::

        data/processed/
          pad/clinical_context/
          isic18/clinical_context/
          ...
    """
    processed = resolve_processed_root(root)
    allow = set(datasets) if datasets else None
    found: list[Path] = []

    for child in sorted(processed.iterdir()):
        if not child.is_dir():
            continue
        if allow is not None and child.name not in allow:
            continue
        variant = child / PROCESSED_VARIANT
        if not variant.is_dir():
            continue
        if (variant / "train.jsonl").exists() or (variant / "dataset_info.json").exists():
            found.append(variant)

    if allow is not None:
        missing = sorted(allow - {p.parent.name for p in found})
        if missing:
            raise FileNotFoundError(
                f"Requested datasets not found under {processed} "
                f"({PROCESSED_VARIANT}): {missing}"
            )

    if not found:
        raise FileNotFoundError(
            f"No datasets found under {processed}. "
            f"Expected folders like pad/{PROCESSED_VARIANT}/train.jsonl"
        )

    return found


def load_processed_dataset(
    path: str | Path,
    split: str = "train",
    image_root: str | Path | None = None,
    validate: bool = True,
) -> Dataset:
    path = Path(path)
    info = load_dataset_info(path) if path.is_dir() else None

    resolved_image_root: Path | None
    if image_root is not None:
        resolved_image_root = Path(image_root)
    elif info and info.image_root:
        resolved_image_root = Path(info.image_root)
    elif path.is_dir():
        candidate = path / "images"
        resolved_image_root = candidate if candidate.exists() else path
    else:
        resolved_image_root = path.parent

    if path.is_file() and path.suffix == ".jsonl":
        rows = _read_jsonl(path)
    elif path.is_dir() and (path / f"{split}.jsonl").exists():
        rows = _read_jsonl(path / f"{split}.jsonl")
    elif path.is_dir() and (path / "dataset_info.json").exists() is False and (path / "state.json").exists():
        ds = load_from_disk(str(path))
        if isinstance(ds, dict):
            if split not in ds:
                raise KeyError(f"Split {split!r} not found in {path}")
            return ds[split]
        return ds
    elif path.is_dir() and (path / split).exists() and (path / split / "state.json").exists():
        return load_from_disk(str(path / split))
    else:
        raise FileNotFoundError(
            f"Could not find processed dataset at {path}. "
            f"Expected {split}.jsonl, a JSONL file, or a Hugging Face saved dataset."
        )

    processed: list[dict[str, Any]] = []
    for row in rows:
        if validate:
            sample = validate_sample(row)
            row = sample.to_trl_dict()
        row = _rewrite_image_paths(row, resolved_image_root)
        if validate:
            Sample.model_validate(row)
        processed.append(row)

    if not processed:
        raise ValueError(f"Dataset at {path} is empty")

    return Dataset.from_list(processed)


def load_multi_processed_dataset(
    root: str | Path,
    split: str = "train",
    *,
    datasets: list[str] | None = None,
    image_root: str | Path | None = None,
    validate: bool = True,
    seed: int | None = None,
) -> Dataset:
    """Load processed ``clinical_context`` splits.

    Pass ``datasets=["pad"]`` for the main study. Omit the filter only when
    concatenating additional experimental sources. Each dataset keeps its own
    ``image_root`` from ``dataset_info.json`` unless a global ``image_root``
    override is provided.
    """
    paths = discover_context_datasets(root, datasets=datasets)
    loaded: list[Dataset] = []

    for path in paths:
        if not (path / f"{split}.jsonl").exists():
            logger.warning("Skipping %s: missing %s.jsonl", path, split)
            continue
        logger.info("Loading %s split from %s", split, path)
        loaded.append(
            load_processed_dataset(
                path,
                split=split,
                image_root=image_root,
                validate=validate,
            )
        )

    if not loaded:
        raise FileNotFoundError(
            f"No {split!r} splits found under {root} ({PROCESSED_VARIANT})"
        )

    combined = concatenate_datasets(loaded) if len(loaded) > 1 else loaded[0]
    logger.info(
        "Combined %s datasets for split=%s -> %s samples",
        len(loaded),
        split,
        len(combined),
    )
    if seed is not None and split == "train":
        combined = combined.shuffle(seed=seed)
    return combined
