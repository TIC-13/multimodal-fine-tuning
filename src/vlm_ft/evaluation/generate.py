from __future__ import annotations

from typing import Any

import torch

from vlm_ft.data.collate import _extract_images_from_messages, _messages_for_chat_template
from vlm_ft.evaluation.config import GenerationConfig
from vlm_ft.evaluation.labels import user_messages


def generate_response(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    cfg: GenerationConfig,
    *,
    max_length: int = 4096,
) -> str:
    user_only = user_messages(messages)
    template_messages = _messages_for_chat_template(user_only)
    images = _extract_images_from_messages(user_only)

    text = processor.apply_chat_template(
        template_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        inputs = processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    except TypeError:
        inputs = processor(
            texts=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        )

    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample or cfg.temperature > 0,
    }
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = cfg.temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, input_len:]
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
