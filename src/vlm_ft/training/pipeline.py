from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from vlm_ft.architectures.base import LoadedModel, ModulePaths, VLMModel
from vlm_ft.architectures.dtype import resolve_torch_dtype
from vlm_ft.architectures.registry import get_model
from vlm_ft.config.training import TrainConfig, TrainableConfig, load_train_config
from vlm_ft.data.dataset import discover_context_datasets, load_multi_processed_dataset
from vlm_ft.data.information_dropout import InformationDropoutDataset
from vlm_ft.training.freeze import apply_trainable_modules, enable_non_lora_trainable
from vlm_ft.training.peft_setup import apply_peft, build_bnb_config
from vlm_ft.training.sft import build_trainer

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a VLM with transformers + PEFT + TRL")
    parser.add_argument("--config", type=str, required=True, help="Path to train YAML config")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


class _TrainingPipeline:
    def __init__(self, config_path: str | Path) -> None:
        self.cfg: TrainConfig = load_train_config(config_path)
        self.architecture: VLMModel = get_model(self.cfg.model.family)
        self.train_dataset: Any | None = None
        self.eval_dataset: Any | None = None
        self.test_dataset: Any | None = None
        self.model: Any | None = None
        self.processor: Any | None = None
        self.module_paths: ModulePaths | None = None

    def run(self) -> Path:
        self._load_datasets()
        self._load_model()
        self._prepare_trainable_params()
        trainer = self._build_trainer()
        return self._fit_and_save(trainer)

    def _load_split(self, split: str) -> Any | None:
        try:
            return load_multi_processed_dataset(
                self.cfg.data.root,
                split=split,
                datasets=self.cfg.data.datasets,
                image_root=self.cfg.data.image_root,
                seed=self.cfg.training.seed if split == "train" else None,
            )
        except FileNotFoundError:
            if split == "train":
                raise
            logger.warning("No %s split found under %s", split, self.cfg.data.root)
            return None

    def _load_datasets(self) -> None:
        paths = discover_context_datasets(
            self.cfg.data.root,
            datasets=self.cfg.data.datasets,
        )
        logger.info(
            "Loading datasets: root=%s -> %s",
            self.cfg.data.root,
            [str(p) for p in paths],
        )
        self.train_dataset = self._load_split("train")
        self.eval_dataset = self._load_split("validation")
        self.test_dataset = self._load_split("test")

        rate = self.cfg.data.information_rate
        if self.train_dataset is not None and rate < 1.0:
            logger.info("Wrapping train dataset with information_rate=%.4f", rate)
            self.train_dataset = InformationDropoutDataset(self.train_dataset, rate)

    def _load_model(self) -> None:
        dtype = resolve_torch_dtype(self.cfg.model.torch_dtype)
        quant_config = build_bnb_config(self.cfg.method, self.cfg.quantization)

        logger.info(
            "Loading model family=%s id=%s method=%s",
            self.cfg.model.family,
            self.cfg.model.model_id,
            self.cfg.method,
        )
        loaded: LoadedModel = self.architecture.load_model_and_processor(
            self.cfg.model.model_id,
            torch_dtype=dtype,
            attn_implementation=self.cfg.model.attn_implementation,
            quantization_config=quant_config,
            trust_remote_code=self.cfg.model.trust_remote_code,
        )
        self.model = loaded.model
        self.processor = loaded.processor
        self.module_paths = self.architecture.module_paths()

    def _prepare_trainable_params(self) -> None:
        assert self.model is not None and self.module_paths is not None

        if self.cfg.method == "full":
            self._prepare_full_finetune()
        else:
            self._prepare_peft()

        if self.cfg.training.gradient_checkpointing and hasattr(
            self.model, "enable_input_require_grads"
        ):
            self.model.enable_input_require_grads()

    def _prepare_full_finetune(self) -> None:
        assert self.model is not None and self.module_paths is not None
        apply_trainable_modules(self.model, self.module_paths, self.cfg.trainable)

    def _prepare_peft(self) -> None:
        assert self.model is not None and self.module_paths is not None
        apply_trainable_modules(
            self.model,
            self.module_paths,
            TrainableConfig(llm=False, projector=False, encoder=False),
        )
        self.model = apply_peft(
            self.model,
            self.cfg.method,
            self.cfg.lora,
            self.architecture,
            self.module_paths,
            self.cfg.trainable,
        )
        enable_non_lora_trainable(self.model, self.module_paths, self.cfg.trainable)

    def _build_trainer(self) -> Any:
        return build_trainer(
            cfg=self.cfg,
            model=self.model,
            processor=self.processor,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

    def _fit_and_save(self, trainer: Any) -> Path:
        output_dir = Path(self.cfg.training.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Starting training -> %s", output_dir)
        trainer.train()
        trainer.save_model(str(output_dir))
        self._save_processor(output_dir)
        self._evaluate_test(trainer, output_dir)

        logger.info("Training complete. Artifacts in %s", output_dir)
        return output_dir

    def _evaluate_test(self, trainer: Any, output_dir: Path) -> None:
        if self.test_dataset is None:
            return
        logger.info("Evaluating best checkpoint on test split")
        metrics = trainer.evaluate(self.test_dataset, metric_key_prefix="test")
        serializable = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in metrics.items()
        }
        metrics_path = output_dir / "test_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
            f.write("\n")
        logger.info("Test metrics: %s", serializable)

    def _save_processor(self, output_dir: Path) -> None:
        if self.processor is not None and hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(str(output_dir))


def run_training(config_path: str | Path) -> Path:
    return _TrainingPipeline(config_path).run()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_training(args.config)


if __name__ == "__main__":
    main()
