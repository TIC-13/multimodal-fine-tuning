# Multimodal clinical context in dermatological VLMs

Research code for studying how **multimodal foundation models** use **clinical context** during adaptation, and whether that adaptation creates **metadata dependence**.

**Central question:** can the model gain from clinical context without losing its **image-only** diagnostic ability when metadata are partially or fully absent?

The main study uses **PAD-UFES-20**, one foundation model, and one adaptation method chosen after a limited pilot. Other datasets in this repository remain available for exploration or possible external replication. Research design: [docs/architecture/project.md](docs/architecture/project.md). Proposal: [docs/proposal/proposal.pdf](docs/proposal/proposal.pdf).

**Training stack:** transformers + PEFT + TRL. Implemented methods: LoRA, QLoRA, full fine-tuning (`llm` / `projector` / `encoder`). LoRA can also target the vision encoder (`lora.encoder: true`). Details: [docs/architecture/training-methods.md](docs/architecture/training-methods.md).

**Model families in the repo:** `medgemma`, `gemma4`, `qwen3_5`. MedGemma and Gemma 4 are gated on Hugging Face — accept the license and authenticate before training.

## Setup

```bash
make install
make install-flash   # optional, needs CUDA
```

Requires Python >= 3.10 and `transformers>=5.2.0`.

## Usage

```bash
make help
make notebook
make train CONFIG=configs/train/medgemma_lora.yaml
make eval EVAL_CONFIG=configs/eval/medgemma.yaml
make lint
make clean
```

Training configs live in `configs/train/`. Training uses the train split, picks the best checkpoint on validation (`eval_loss`), then evaluates once on test (`outputs/.../test_metrics.json`). Classification metrics on the test split (generation + canonical **code** parsing) are available via `make eval EVAL_CONFIG=configs/eval/medgemma.yaml`.

For the **main study**, restrict data to PAD and set `information_rate` to the regime under test:

```yaml
data:
  root: data
  datasets: [pad]
  information_rate: 1.0   # 0 = image-only, 1 = observed context, (0, 1) = stochastic masking
```

## Data

Prepare datasets with `make notebook` using the notebooks in `notebooks/prepare/`. Processed data goes under `data/processed/<name>/clinical_context/` (full prompt version on disk):

```
data/processed/
  <name>/clinical_context/
```

| Source | Role |
|---|---|
| **PAD-UFES-20** (`pad`) | Primary dataset for the main study |
| MILK10k (`milk10k`) | Possible external replication, if labels, clinical images, and metadata are compatible |
| ISIC 2018, HC, Derm1M, DDI, SD-198 | Kept in the repo as experimental / exploration resources |

Canonical sample / prompt / label builders live in `src/vlm_ft/data/canonical.py`. Each dataset has `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `dataset_info.json` (with `image_root` pointing at `data/datasets/...`). The `data/` folder is gitignored. Format details: [docs/architecture/dataset-format.md](docs/architecture/dataset-format.md).

If `datasets` is omitted, the trainer concatenates every processed folder under `data/processed/`. That mix is useful for exploration; it is **not** the main-study protocol.

`information_rate` applies **only during training**: each clinical `- Field: value` line in the user prompt is kept independently with that probability (eval/test stay full on disk). At `0`, all metadata is stripped to an image-only diagnosis prompt. Masking is part of the method; it is not presented as the main scientific novelty.

PAD splits are grouped by patient/lesion. The evaluation protocol (metadata availability grid, repeated masks, field/group ablations, balanced accuracy, macro-F1, bootstrap CIs) is specified in [docs/architecture/project.md](docs/architecture/project.md).

## Add a model

1. Add an implementation in `src/vlm_ft/architectures/models/`
2. Register it in `src/vlm_ft/architectures/registry.py`
3. Add a config in `configs/train/`

Outputs are written to `outputs/` (gitignored).
