"""Helpers shared by ``notebooks/prepare/*``: native source loaders, splits, EDA."""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from vlm_ft.data.canonical import is_missing
from vlm_ft.data.convert import write_dataset_info, write_jsonl
from vlm_ft.data.dataset import load_processed_dataset
from vlm_ft.data.schema import DatasetInfo, validate_sample

SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_SEED = 42

ISIC18_TASK3_COLUMNS = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
ISIC18_SPLIT_LAYOUT = {
    "train": (
        "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
        "ISIC2018_Task3_Training_Input",
        "ISIC2018_Task3_Training_LesionGroupings.csv",
    ),
    "validation": (
        "ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv",
        "ISIC2018_Task3_Validation_Input",
        None,
    ),
    "test": (
        "ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
        "ISIC2018_Task3_Test_Input",
        None,
    ),
}

MILK10K_CLASS_COLUMNS = (
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
)


def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find repo root (src/) from {start or Path.cwd()}")


def rel_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def extract_zip_images(image_dir: Path) -> int:
    """Flatten ``*.zip`` members into ``image_dir``. Skip names that already exist."""
    image_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for zip_path in sorted(image_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                dest = image_dir / Path(info.filename).name
                if dest.exists():
                    continue
                dest.write_bytes(zf.read(info.filename))
                written += 1
    return written


def split_by_group(
    df: pd.DataFrame,
    *,
    group_col: str | list[str] | tuple[str, ...],
    label_col: str | None,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = SPLIT_SEED,
) -> dict[str, pd.DataFrame]:
    """Assign groups to train/validation/test.

    When ``label_col`` is set, groups are stratified by each group's first label.
    When it is ``None``, groups are shuffled as a single pool (better for open-vocab
    sources with thousands of singleton classes).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    out = df.copy()
    if isinstance(group_col, (list, tuple)):
        missing = [c for c in group_col if c not in out.columns]
        if missing:
            raise KeyError(missing)
        key_col = "_split_group"
        out[key_col] = out[list(group_col)].astype(str).agg("\t".join, axis=1)
    else:
        if group_col not in out.columns:
            raise KeyError(group_col)
        key_col = group_col

    if label_col is None:
        grouped = pd.Series("all", index=out[key_col].drop_duplicates())
    else:
        if label_col not in out.columns:
            raise KeyError(label_col)
        grouped = out.groupby(key_col, sort=False)[label_col].agg(lambda s: s.iloc[0])
    label_to_groups: dict[Any, list[Any]] = {}
    for group_id, label in grouped.items():
        label_to_groups.setdefault(label, []).append(group_id)

    rng = random.Random(seed)
    assigned: dict[Any, str] = {}
    for groups in label_to_groups.values():
        groups = list(groups)
        rng.shuffle(groups)
        n = len(groups)
        if n == 1:
            assigned[groups[0]] = "train"
            continue
        n_train = max(1, int(n * ratios[0]))
        n_val = int(n * ratios[1])
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        assigned.update(dict.fromkeys(groups[:n_train], "train"))
        assigned.update(dict.fromkeys(groups[n_train : n_train + n_val], "validation"))
        assigned.update(dict.fromkeys(groups[n_train + n_val :], "test"))

    out["split"] = out[key_col].map(assigned)
    missing = out["split"].isna().sum()
    if missing:
        raise ValueError(f"{missing} rows did not receive a split assignment")
    if key_col == "_split_group":
        out = out.drop(columns=[key_col])

    return {split: out.loc[out["split"] == split].copy() for split in SPLITS}


def missing_rate(series: pd.Series) -> float:
    return float(series.map(is_missing).mean())


def print_source_eda(
    df: pd.DataFrame,
    *,
    label_col: str,
    metadata_cols: Iterable[str] = (),
    split_col: str | None = "split",
    extra: Mapping[str, pd.Series] | None = None,
) -> None:
    print(f"rows={len(df):,}  columns={len(df.columns)}")
    if split_col and split_col in df.columns:
        print("\nSplit sizes:")
        print(df[split_col].value_counts().reindex(list(SPLITS)).to_string())
        overlap = (
            df.groupby(split_col)[label_col].nunique()
            if label_col in df.columns
            else None
        )
        if overlap is not None:
            print("\nUnique labels per split:")
            print(overlap.reindex(list(SPLITS)).to_string())

    print(f"\nLabel `{label_col}` ({df[label_col].nunique(dropna=True)} unique):")
    print(df[label_col].value_counts(dropna=False).head(25).to_string())

    cols = [c for c in metadata_cols if c in df.columns]
    if cols:
        print("\nMetadata missing / UNK rate:")
        for col in cols:
            print(f"  {col}: {missing_rate(df[col]) * 100:5.1f}%")

    if extra:
        print("\nExtra:")
        for name, series in extra.items():
            print(f"  {name}:")
            print("   ", series.to_string().replace("\n", "\n    "))


def count_existing_images(df: pd.DataFrame, rel_col: str, root: Path, *, limit: int | None = None) -> tuple[int, int]:
    subset = df if limit is None else df.head(limit)
    ok = 0
    total = 0
    for rel in subset[rel_col].astype(str):
        total += 1
        if (root / rel).is_file():
            ok += 1
    return ok, total


def decode_one_hot(row: Mapping[str, Any], columns: Iterable[str]) -> str:
    hits = [col for col in columns if float(row.get(col, 0) or 0) >= 0.5]
    if len(hits) != 1:
        raise ValueError(f"Expected a single one-hot class, got {hits} from {dict(row)}")
    return hits[0]


def load_pad(root: Path) -> pd.DataFrame:
    extracted = extract_zip_images(root / "images")
    if extracted:
        print(f"Extracted {extracted} images from zip archives into {root / 'images'}")
    df = pd.read_csv(root / "metadata.csv")
    df["image_rel"] = "images/" + df["img_id"].astype(str)
    return df


def load_hc(root: Path) -> pd.DataFrame:
    from vlm_ft.data.canonical import hc_map_diagnosis, hc_pick_raw_diagnosis

    df = pd.read_csv(root / "lesions.csv")
    df = df.loc[df["imageCropped"].notna()].copy()
    df["imageCropped"] = df["imageCropped"].astype(str).str.strip()
    df = df.loc[df["imageCropped"].ne("") & df["imageCropped"].str.lower().ne("nan")]
    pairs = [hc_map_diagnosis(hc_pick_raw_diagnosis(row)) for row in df.to_dict("records")]
    df["diagnosis"] = [item[0] for item in pairs]
    df["code"] = [item[1] for item in pairs]
    df["image_rel"] = "images/" + df["imageCropped"]
    exists = df["image_rel"].map(lambda rel: (root / str(rel)).is_file())
    dropped = int((~exists).sum())
    if dropped:
        print(f"Dropped {dropped} rows with missing cropped images")
    return df.loc[exists].reset_index(drop=True)


def load_isic18(root: Path) -> dict[str, pd.DataFrame]:
    from vlm_ft.data.canonical import ISIC18_TASK3_CLASSES

    splits: dict[str, pd.DataFrame] = {}
    for split, (gt_rel, image_dir, groups_rel) in ISIC18_SPLIT_LAYOUT.items():
        gt = pd.read_csv(root / gt_rel)
        codes = [decode_one_hot(row, ISIC18_TASK3_COLUMNS) for row in gt.to_dict("records")]
        mapped = [ISIC18_TASK3_CLASSES[code] for code in codes]
        out = pd.DataFrame(
            {
                "image": gt["image"].astype(str),
                "task3_code": codes,
                "diagnosis": [item[0] for item in mapped],
                "code": [item[1] for item in mapped],
                "malignancy": [item[2] for item in mapped],
                "image_rel": [f"{image_dir}/{stem}.jpg" for stem in gt["image"].astype(str)],
                "split": split,
            }
        )
        if groups_rel is not None:
            groups = pd.read_csv(root / groups_rel)
            out = out.merge(
                groups.rename(columns={"image": "image"}),
                on="image",
                how="left",
            )
        else:
            out["diagnosis_confirm_type"] = pd.NA
        splits[split] = out
    return splits


def load_milk10k(root: Path) -> pd.DataFrame:
    from vlm_ft.data.canonical import MILK10K_CLASSES

    gt = pd.read_csv(root / "MILK10k_Training_GroundTruth.csv")
    meta = pd.read_csv(root / "MILK10k_Training_Metadata.csv")
    supplement = pd.read_csv(root / "MILK10k_Training_Supplement.csv")
    gt["task_code"] = [decode_one_hot(row, MILK10K_CLASS_COLUMNS) for row in gt.to_dict("records")]
    mapped = gt["task_code"].map(lambda code: MILK10K_CLASSES[code])
    gt["diagnosis"] = [item[0] for item in mapped]
    gt["code"] = [item[1] for item in mapped]
    gt["malignancy"] = [item[2] for item in mapped]
    df = meta.merge(gt[["lesion_id", "task_code", "diagnosis", "code", "malignancy"]], on="lesion_id")
    df = df.merge(supplement, on="isic_id", how="left")
    df["image_rel"] = (
        "MILK10k_Training_Input/" + df["lesion_id"].astype(str) + "/" + df["isic_id"].astype(str) + ".jpg"
    )
    return df


def load_derm1m_labeled(root: Path, *, limit: int | None = None) -> pd.DataFrame:
    from vlm_ft.data.canonical import DERM1M_SKIP_DIAGNOSES, derm1m_is_missing

    path = root / "Derm1M_v2_pretrain.csv"
    usecols = [
        "filename",
        "disease_label",
        "age",
        "gender",
        "body_location",
        "symptoms",
        "skin_concept",
        "source",
        "source_type",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in set(usecols))
    label = df["disease_label"].astype(str).str.strip().str.lower()
    df = df.loc[~label.isin(DERM1M_SKIP_DIAGNOSES) & label.ne("") & ~label.map(derm1m_is_missing)].copy()
    df["disease_label"] = df["disease_label"].astype(str).str.strip()
    df["image_rel"] = "images/" + df["filename"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["filename"], keep="first")
    if limit is not None:
        df = df.head(int(limit)).copy()
    return df.reset_index(drop=True)


def require_source(root: Path, *, expected: str) -> None:
    if not root.exists():
        raise FileNotFoundError(
            f"{root} is not present. Current datasets under data/datasets/: {expected}"
        )


def rows_to_samples(
    df: pd.DataFrame,
    data_root: Path,
    *,
    prompt_and_label,
    image_rel,
) -> list[dict[str, Any]]:
    from vlm_ft.data.canonical import build_sample

    samples: list[dict[str, Any]] = []
    skipped = 0
    for row in df.to_dict("records"):
        rel = image_rel(row)
        if not (data_root / rel).is_file():
            skipped += 1
            continue
        prompt, label = prompt_and_label(row)
        samples.append(build_sample(image_rel=rel, prompt=prompt, assistant_label=label))
    if skipped:
        print(f"skipped {skipped} rows with missing images")
    return samples


def write_processed_dataset(
    *,
    name: str,
    out_dir: Path,
    split_samples: Mapping[str, list[dict[str, Any]]],
    image_root_rel: str,
    source_rel: str | None = None,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in SPLITS:
        samples = split_samples[split]
        counts[split] = write_jsonl(samples, out_dir / f"{split}.jsonl")
        print(f"{name} {split}: {counts[split]} samples")
    info = DatasetInfo(
        name=name,
        num_samples=counts["train"],
        splits=list(SPLITS),
        image_root=image_root_rel,
        source=source_rel or image_root_rel,
    )
    write_dataset_info(out_dir / "dataset_info.json", info)
    print(f"Wrote dataset to {out_dir}")
    print(json.dumps(info.model_dump(), indent=2))
    return counts


def preview_processed(out_dir: Path, title: str) -> None:
    preview_path = out_dir / "train.jsonl"
    with preview_path.open("r", encoding="utf-8") as f:
        sample = validate_sample(json.loads(f.readline()))

    print(f"=== {title} ===")
    print("User prompt:\n")
    for part in sample.messages[0].content:
        if part.type == "text":
            print(part.text)
        else:
            print(f"[image: {part.image}]")

    print("\nAssistant label:\n")
    print(sample.messages[1].content[0].text)

    train_ds = load_processed_dataset(out_dir, split="train")
    val_ds = load_processed_dataset(out_dir, split="validation")
    test_ds = load_processed_dataset(out_dir, split="test")
    print(f"\nLoader check -> train: {len(train_ds)}, validation: {len(val_ds)}, test: {len(test_ds)}")
