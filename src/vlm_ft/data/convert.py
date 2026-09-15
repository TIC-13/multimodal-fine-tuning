from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from PIL import Image

from vlm_ft.data.schema import DatasetInfo, Sample, validate_sample


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return value.strip("_") or "sample"


def save_image(image: Any, dest_dir: Path, stem: str) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{stem}.jpg"

    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        rgb.save(out_path, format="JPEG", quality=95)
    elif isinstance(image, (str, Path)):
        src = Path(image)
        with Image.open(src) as im:
            im.convert("RGB").save(out_path, format="JPEG", quality=95)
    elif isinstance(image, dict) and "bytes" in image:
        from io import BytesIO

        with Image.open(BytesIO(image["bytes"])) as im:
            im.convert("RGB").save(out_path, format="JPEG", quality=95)
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    return str(Path("images") / out_path.name)


def messages_from_llava_style(
    conversations: list[dict[str, Any]],
    image_path: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    image_attached = False

    for turn in conversations:
        role_raw = turn.get("from") or turn.get("role")
        value = turn.get("value") or turn.get("content") or ""
        if role_raw in {"human", "user"}:
            role = "user"
        elif role_raw in {"gpt", "assistant"}:
            role = "assistant"
        elif role_raw == "system":
            role = "system"
        else:
            raise ValueError(f"Unknown conversation role: {role_raw!r}")

        content: list[dict[str, Any]] = []
        text = value
        if isinstance(value, str):
            text = (
                value.replace("<image>", "")
                .replace("<image>\n", "")
                .replace("\n<image>", "")
                .strip()
            )
        if role == "user" and image_path and not image_attached:
            content.append({"type": "image", "image": image_path})
            image_attached = True
        if text:
            content.append({"type": "text", "text": text})
        elif role == "user" and content:
            pass
        else:
            content.append({"type": "text", "text": str(value)})

        messages.append({"role": role, "content": content})

    return messages


def normalize_trl_messages(messages: list[dict[str, Any]], image_paths: list[str] | None = None) -> list[dict[str, Any]]:
    image_paths = list(image_paths or [])
    image_idx = 0
    normalized: list[dict[str, Any]] = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        if isinstance(content, str):
            parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
            if role == "user" and image_idx < len(image_paths):
                parts.insert(0, {"type": "image", "image": image_paths[image_idx]})
                image_idx += 1
            normalized.append({"role": role, "content": parts})
            continue

        parts = []
        for part in content:
            ptype = part.get("type")
            if ptype in {"image", "image_url"}:
                if image_idx < len(image_paths):
                    parts.append({"type": "image", "image": image_paths[image_idx]})
                    image_idx += 1
                elif part.get("image"):
                    parts.append({"type": "image", "image": part["image"]})
                elif isinstance(part.get("image_url"), dict) and part["image_url"].get("url"):
                    parts.append({"type": "image", "image": part["image_url"]["url"]})
                else:
                    raise ValueError("Image content part missing resolvable image path")
            elif ptype == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            else:
                if "text" in part:
                    parts.append({"type": "text", "text": part["text"]})
                elif "image" in part:
                    parts.append({"type": "image", "image": part["image"]})
        normalized.append({"role": role, "content": parts})

    return normalized


def write_jsonl(samples: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            validate_sample(sample)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_dataset_info(path: Path, info: DatasetInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info.model_dump(), f, indent=2, ensure_ascii=False)
        f.write("\n")


def convert_hf_dataset_to_canonical(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    name: str,
    source: str | None = None,
    split: str = "train",
    row_to_sample: Callable[[dict[str, Any], Path, int], dict[str, Any]],
    limit: int | None = None,
) -> DatasetInfo:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        sample = row_to_sample(row, images_dir, idx)
        Sample.model_validate(sample)
        samples.append(sample)

    count = write_jsonl(samples, output_dir / f"{split}.jsonl")
    info = DatasetInfo(
        name=name,
        num_samples=count,
        splits=[split],
        image_root=str(output_dir),
        source=source,
    )
    write_dataset_info(output_dir / "dataset_info.json", info)
    return info


def default_llava_row_converter(
    image_key: str = "image",
    conversations_key: str = "conversations",
) -> Callable[[dict[str, Any], Path, int], dict[str, Any]]:
    def _convert(row: dict[str, Any], images_dir: Path, idx: int) -> dict[str, Any]:
        stem = f"{idx:06d}_{_slug(str(row.get('id', uuid4().hex[:8])))}"
        rel_image = save_image(row[image_key], images_dir, stem)
        messages = messages_from_llava_style(row[conversations_key], image_path=rel_image)
        return {"messages": messages}

    return _convert


def _extract_embedded_images(messages: list[dict[str, Any]]) -> list[Any]:
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") not in {"image", "image_url"}:
                continue
            if isinstance(part.get("image"), (Image.Image, dict, str, Path)):
                images.append(part["image"])
            elif isinstance(part.get("image_url"), dict) and part["image_url"].get("url"):
                images.append(part["image_url"]["url"])
    return images


def default_trl_row_converter(
    messages_key: str = "messages",
    images_key: str = "images",
) -> Callable[[dict[str, Any], Path, int], dict[str, Any]]:
    def _convert(row: dict[str, Any], images_dir: Path, idx: int) -> dict[str, Any]:
        messages = row[messages_key]
        images = row.get(images_key) or []
        if images and not isinstance(images, list):
            images = [images]
        if not images:
            images = _extract_embedded_images(messages)

        rel_paths: list[str] = []
        for j, image in enumerate(images):
            if isinstance(image, str) and image.startswith(("http://", "https://")):
                rel_paths.append(image)
                continue
            stem = f"{idx:06d}_{j:02d}"
            rel_paths.append(save_image(image, images_dir, stem))
        normalized = normalize_trl_messages(messages, image_paths=rel_paths)
        return {"messages": normalized}

    return _convert
