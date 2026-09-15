from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


Method = Literal["lora", "qlora", "full"]
ModelFamily = Literal["medgemma", "gemma4", "qwen3_5"]
SUPPORTED_FAMILIES = {"medgemma", "gemma4", "qwen3_5"}

@dataclass
class ModelConfig:
    family: ModelFamily
    model_id: str
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = "flash_attention_2"
    trust_remote_code: bool = True

@dataclass
class TrainableConfig:
    llm: bool = True
    projector: bool = False
    encoder: bool = False


@dataclass
class LoraConfigArgs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "auto"
    encoder: bool = False
    encoder_target_modules: str | list[str] = "auto"
    bias: str = "none"

@dataclass
class QuantizationConfig:
    bits: int = 4
    quant_type: str = "nf4"
    use_double_quant: bool = True
    compute_dtype: str = "bfloat16"


@dataclass
class DataConfig:
    """Dataset loading.

    Discovers ``{dataset}/clinical_context/`` under ``root``. The main study
    should set ``datasets: [pad]``; omit the filter only for exploration::

        data:
          root: data
          datasets: [pad]
          information_rate: 1.0   # 0=image-only, 1=observed, (0,1)=stochastic masking
    """

    root: str
    datasets: list[str] | None = None
    max_length: int = 4096
    image_root: str | None = None
    information_rate: float = 1.0


@dataclass
class TrainingArgsConfig:
    output_dir: str = "outputs/run"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int | None = None
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    report_to: str = "tensorboard"
    seed: int = 42
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    push_to_hub: bool = False


@dataclass
class TrainConfig:
    model: ModelConfig
    method: Method
    data: DataConfig
    trainable: TrainableConfig = field(default_factory=TrainableConfig)
    lora: LoraConfigArgs = field(default_factory=LoraConfigArgs)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingArgsConfig = field(default_factory=TrainingArgsConfig)


def _merge_dataclass(cls: type, raw: dict[str, Any] | None, defaults: Any | None = None):
    base = {}
    if defaults is not None:
        base.update(defaults.__dict__)
    if raw:
        base.update(raw)
    return cls(**base)


def load_train_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a mapping")

    method = raw.get("method")
    if method not in {"lora", "qlora", "full"}:
        raise ValueError(f"method must be lora|qlora|full, got {method!r}")

    model_raw = raw.get("model") or {}
    family = model_raw.get("family")
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported model.family: {family!r}")

    cfg = TrainConfig(
        model=_merge_dataclass(ModelConfig, model_raw),
        method=method,
        trainable=_merge_dataclass(TrainableConfig, raw.get("trainable")),
        lora=_merge_dataclass(LoraConfigArgs, raw.get("lora")),
        quantization=_merge_dataclass(QuantizationConfig, raw.get("quantization")),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        training=_merge_dataclass(TrainingArgsConfig, raw.get("training")),
    )
    _validate_train_config(cfg)
    return cfg


def _validate_train_config(cfg: TrainConfig) -> None:
    if cfg.lora.encoder and cfg.trainable.encoder:
        raise ValueError(
            "lora.encoder and trainable.encoder cannot both be true. "
            "Use lora.encoder for adapters on the vision encoder, or trainable.encoder "
            "to fully unfreeze encoder weights."
        )
    if cfg.method == "full" and cfg.lora.encoder:
        raise ValueError("lora.encoder is only valid when method is lora or qlora")
    rate = cfg.data.information_rate
    if not isinstance(rate, (int, float)) or rate < 0.0 or rate > 1.0:
        raise ValueError(f"data.information_rate must be in [0, 1], got {rate!r}")

