from __future__ import annotations

from pathlib import Path
from typing import Any

from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from vlm_ft.architectures.dtype import resolve_torch_dtype
from vlm_ft.evaluation.config import EvalModelConfig


def _is_peft_checkpoint(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def load_eval_model(cfg: EvalModelConfig) -> tuple[Any, Any]:
    dtype = resolve_torch_dtype(cfg.torch_dtype)

    checkpoint = Path(cfg.checkpoint) if cfg.checkpoint else None
    load_path = str(checkpoint) if checkpoint and not _is_peft_checkpoint(checkpoint) else cfg.model_id

    kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "device_map": "auto",
        "torch_dtype": dtype,
    }
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation

    try:
        model = AutoModelForImageTextToText.from_pretrained(load_path, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        model = AutoModelForImageTextToText.from_pretrained(load_path, **kwargs)

    if checkpoint and _is_peft_checkpoint(checkpoint):
        model = PeftModel.from_pretrained(model, str(checkpoint))

    processor_source = str(checkpoint) if checkpoint else cfg.model_id
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=cfg.trust_remote_code)

    model.eval()
    return model, processor
