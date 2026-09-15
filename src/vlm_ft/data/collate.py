from __future__ import annotations

from typing import Any, Callable

from PIL import Image


def _load_image(path_or_img: Any) -> Image.Image:
    if isinstance(path_or_img, Image.Image):
        return path_or_img.convert("RGB")
    return Image.open(path_or_img).convert("RGB")


def _extract_images_from_messages(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image":
                images.append(_load_image(part["image"]))
    return images


def _messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            rendered.append({"role": message["role"], "content": content})
            continue
        new_content = []
        for part in content:
            if part.get("type") == "image":
                new_content.append({"type": "image"})
            elif part.get("type") == "text":
                new_content.append({"type": "text", "text": part.get("text", "")})
        rendered.append({"role": message["role"], "content": new_content})
    return rendered


def build_collate_fn(processor: Any, family: str, max_length: int = 4096) -> Callable:
    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        all_images: list[list[Image.Image]] = []

        for example in examples:
            messages = example["messages"]
            images = _extract_images_from_messages(messages)
            template_messages = _messages_for_chat_template(messages)
            text = processor.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            all_images.append(images)

        flat_images = [img for images in all_images for img in images]

        try:
            batch = processor(
                text=texts,
                images=flat_images if flat_images else None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        except TypeError:
            batch = processor(
                texts,
                images=flat_images if flat_images else None,
                return_tensors="pt",
                padding=True,
            )

        labels = batch["input_ids"].clone()
        pad_token_id = getattr(processor, "tokenizer", processor).pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None and hasattr(processor, "tokenizer"):
            image_token_id = getattr(processor.tokenizer, "image_token_id", None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch

    collate_fn.family = family
    return collate_fn
