from __future__ import annotations

import argparse
from pathlib import Path

from vlm_ft.training.pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a VLM from a YAML config")
    parser.add_argument("config", type=Path, help="Path to train YAML config")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
