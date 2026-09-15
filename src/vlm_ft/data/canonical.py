"""Canonical skin-lesion VLM sample model.

Source tables keep dataset-specific column names; adapters normalize into this
model before writing JSONL.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Mapping

# Assistant gold is a JSON string. When the source is explicit about
# benign/malignant: {"malignancy": "...", "diagnosis": "<name>", "code": "<CODE>"}.
# Otherwise: {"diagnosis": "<name>", "code": "<CODE>"}.

MALIGNANCY_VALUES = frozenset({"benign", "malignant"})

# Closed codes observed across PAD (primary) and the other prepare adapters.
DIAGNOSTIC_CODES: dict[str, str] = {
    # PAD (code → name; inverted below for name→code lookups)
    "basal cell carcinoma": "BCC",
    "squamous cell carcinoma": "SCC",
    "melanoma": "MEL",
    "nevus": "NEV",
    "seborrheic keratosis": "SEK",
    "actinic keratosis": "ACK",
    # ISIC18 extras
    "pigmented benign keratosis": "BKL",
    "vascular lesion": "VASC",
    "dermatofibroma": "DF",
    # DDI extras / hyphenated aliases
    "melanocytic nevus": "NEV",
    "dysplastic nevus": "DYSN",
    "blue nevus": "BN",
    "epidermal nevus": "EN",
    "nevus lipomatosus superficialis": "NLS",
    "atypical spindle cell nevus of reed": "ASCNR",
    "pigmented spindle cell nevus of reed": "PSCNR",
    "solar lentigo": "SL",
    "benign other": "BEN_OTH",
    "inflammatory or infectious disease": "INF",
    "malignant other": "MAL_OTH",
}

PAD_DIAGNOSTIC_NAMES: dict[str, str] = {
    "BCC": "basal cell carcinoma",
    "SCC": "squamous cell carcinoma",
    "MEL": "melanoma",
    "NEV": "nevus",
    "SEK": "seborrheic keratosis",
    "ACK": "actinic keratosis",
}

ISIC18_DIAGNOSTIC_CODES: dict[str, str] = {
    "nevus": "NEV",
    "melanoma": "MEL",
    "pigmented benign keratosis": "BKL",
    "basal cell carcinoma": "BCC",
    "squamous cell carcinoma": "SCC",
    "vascular lesion": "VASC",
    "actinic keratosis": "ACK",
    "dermatofibroma": "DF",
}

DDI_DIAGNOSTIC_CODES: dict[str, str] = {
    "melanoma": "MEL",
    "melanocytic-nevus": "NEV",
    "dysplastic-nevus": "DYSN",
    "blue-nevus": "BN",
    "epidermal-nevus": "EN",
    "nevus-lipomatosus-superficialis": "NLS",
    "atypical-spindle-cell-nevus-of-reed": "ASCNR",
    "pigmented-spindle-cell-nevus-of-reed": "PSCNR",
    "basal-cell-carcinoma": "BCC",
    "squamous-cell-carcinoma": "SCC",
    "seborrheic-keratosis": "SEK",
    "solar-lentigo": "SL",
    "dermatofibroma": "DF",
}

# ISIC 2018 Task 3 one-hot column → (diagnosis, code, malignancy or None).
ISIC18_TASK3_CLASSES: dict[str, tuple[str, str, str | None]] = {
    "MEL": ("melanoma", "MEL", "malignant"),
    "NV": ("nevus", "NEV", "benign"),
    "BCC": ("basal cell carcinoma", "BCC", "malignant"),
    "AKIEC": ("actinic keratosis", "ACK", None),
    "BKL": ("pigmented benign keratosis", "BKL", "benign"),
    "DF": ("dermatofibroma", "DF", "benign"),
    "VASC": ("vascular lesion", "VASC", "benign"),
}

# MILK10k one-hot column → (diagnosis, code, malignancy or None).
MILK10K_CLASSES: dict[str, tuple[str, str, str | None]] = {
    "AKIEC": ("actinic keratosis", "ACK", None),
    "BCC": ("basal cell carcinoma", "BCC", "malignant"),
    "BEN_OTH": ("benign other", "BEN_OTH", "benign"),
    "BKL": ("pigmented benign keratosis", "BKL", "benign"),
    "DF": ("dermatofibroma", "DF", "benign"),
    "INF": ("inflammatory or infectious disease", "INF", None),
    "MAL_OTH": ("malignant other", "MAL_OTH", "malignant"),
    "MEL": ("melanoma", "MEL", "malignant"),
    "NV": ("nevus", "NEV", "benign"),
    "SCCKA": ("squamous cell carcinoma", "SCC", "malignant"),
    "VASC": ("vascular lesion", "VASC", "benign"),
}

HC_CLASS_NAMES = {
    "BCC": "basal cell carcinoma",
    "SCC": "squamous cell carcinoma",
    "MEL": "melanoma",
    "NEV": "nevus",
    "ACK": "actinic keratosis",
    "BKL": "pigmented benign keratosis",
    "DF": "dermatofibroma",
    "VASC": "vascular lesion",
}

DERM1M_SKIP_DIAGNOSES = frozenset({"no definitive diagnosis"})
DERM1M_MISSING_PLACEHOLDERS = frozenset(
    {
        "",
        "nan",
        "none",
        "unk",
        "no age information",
        "no gender information",
        "no body location information",
        "no symptom information",
        "no visual concepts",
    }
)

FITZPATRICK_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
FITZPATRICK_GROUPS = {12: "I-II", 34: "III-IV", 56: "V-VI"}

# English prompt field labels used in Patient / Lesion context sections.
PATIENT_FIELD_LABELS = (
    "Age",
    "Gender",
    "Fitzpatrick skin type",
    "Smoking",
    "Alcohol use",
    "Father ancestry",
    "Mother ancestry",
    "Pesticide exposure",
    "Piped water at home",
    "Sewage system at home",
    "History of skin cancer",
    "History of other cancer",
)

LESION_FIELD_LABELS = (
    "Body region",
    "Diameter",
    "Itching",
    "Growing",
    "Pain",
    "Recent change",
    "Bleeding",
    "Elevation",
    "Biopsied",
    "Image type",
    "Melanocytic",
    "Diagnosis confirmation",
    "Clinical description",
    "Symptoms",
    "Visual concepts",
)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        if math.isnan(value):
            return True
    except TypeError:
        pass
    if isinstance(value, str) and value.strip().upper() in {"", "UNK", "NAN", "NONE", "<NA>"}:
        return True
    return False


def derm1m_is_missing(value: Any) -> bool:
    if is_missing(value):
        return True
    return str(value).strip().lower() in DERM1M_MISSING_PLACEHOLDERS


def fmt_bool(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "yes", "1"}:
            return "yes"
        if value in {"false", "no", "0"}:
            return "no"
        if value == "unk":
            return "unknown"
    if bool(value):
        return "yes"
    return "no"


def fmt_text(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    return text.lower() if text.isupper() and text.isalpha() else text


def pretty_disease_name(value: str) -> str:
    text = re.sub(r"-+", "-", str(value).strip())
    text = text.replace("--", "-").replace("-", " ").replace("(", "").replace(")", "")
    return " ".join(text.split()).lower()


def resolve_malignancy(raw: Any) -> str | None:
    """Return benign|malignant only when the source is explicit; else omit."""
    if isinstance(raw, bool):
        return "malignant" if raw else "benign"
    if is_missing(raw):
        return None
    if isinstance(raw, (int, float)):
        return "malignant" if bool(raw) else "benign"
    label = str(raw).strip().lower()
    if label in {"true", "1", "yes"}:
        return "malignant"
    if label in {"false", "0", "no"}:
        return "benign"
    if label in MALIGNANCY_VALUES:
        return label
    return None


def json_response_instruction(*, include_malignancy: bool) -> str:
    if include_malignancy:
        schema = '{"malignancy": "benign"|"malignant", "diagnosis": "<name>", "code": "<CODE>"}'
    else:
        schema = '{"diagnosis": "<name>", "code": "<CODE>"}'
    return f"Respond with JSON only using this schema: {schema}."


def add_line(lines: list[str], label: str, value: Any) -> None:
    text = fmt_text(value) if not isinstance(value, bool) else None
    if text is None and not isinstance(value, bool):
        return
    if isinstance(value, bool) or str(value).strip().lower() in {"true", "false", "yes", "no", "unk"}:
        lines.append(f"- {label}: {fmt_bool(value)}")
    else:
        lines.append(f"- {label}: {text}")


def build_clinical_prompt(
    *,
    patient_lines: list[str],
    lesion_lines: list[str] | None = None,
    include_malignancy: bool,
) -> str:
    """Canonical user prompt: intro + optional Patient/Lesion sections + JSON schema."""
    lesion_lines = lesion_lines or []
    if patient_lines or lesion_lines:
        sections: list[str] = [
            "Analyze this skin lesion image using the clinical information below.",
        ]
    else:
        sections = ["Analyze this skin lesion image.", ""]
    if patient_lines:
        sections.extend(["Patient context:", *patient_lines, ""])
    if lesion_lines:
        sections.extend(["Lesion context:", *lesion_lines, ""])
    sections.append(json_response_instruction(include_malignancy=include_malignancy))
    return "\n".join(sections)


def build_assistant_label(
    *,
    diagnosis: str,
    code: str,
    malignancy: str | None = None,
) -> str:
    payload: dict[str, str] = {"diagnosis": diagnosis, "code": code}
    if malignancy is not None:
        payload = {"malignancy": malignancy, "diagnosis": diagnosis, "code": code}
    return json.dumps(payload, ensure_ascii=False)


def build_sample(
    *,
    image_rel: str,
    prompt: str,
    assistant_label: str,
) -> dict[str, Any]:
    """Canonical JSONL sample validated by ``vlm_ft.data.schema.Sample``."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rel},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_label},
                ],
            },
        ]
    }


def image_rel_path(img_id: str) -> str:
    return f"images/{img_id}"


# Dataset adapters: current on-disk CSV schemas → canonical fields.


def pad_prompt_and_label(row: Mapping[str, Any]) -> tuple[str, str]:
    """PAD: ``img_id``, ``diagnostic``; no ``benign_malignant`` column in current CSV."""
    patient_lines: list[str] = []
    lesion_lines: list[str] = []

    add_line(patient_lines, "Age", row.get("age"))
    if not is_missing(row.get("gender")):
        patient_lines.append(f"- Gender: {str(row['gender']).strip().lower()}")

    if not is_missing(row.get("fitspatrick")):
        try:
            level = int(float(row["fitspatrick"]))  # type: ignore[arg-type]
            fitz = FITZPATRICK_ROMAN.get(level, str(level))
            patient_lines.append(f"- Fitzpatrick skin type: {fitz}")
        except (TypeError, ValueError):
            pass

    add_line(patient_lines, "Smoking", row.get("smoke"))
    add_line(patient_lines, "Alcohol use", row.get("drink"))
    add_line(patient_lines, "Father ancestry", row.get("background_father"))
    add_line(patient_lines, "Mother ancestry", row.get("background_mother"))
    add_line(patient_lines, "Pesticide exposure", row.get("pesticide"))
    add_line(patient_lines, "Piped water at home", row.get("has_piped_water"))
    add_line(patient_lines, "Sewage system at home", row.get("has_sewage_system"))
    add_line(patient_lines, "History of skin cancer", row.get("skin_cancer_history"))
    add_line(patient_lines, "History of other cancer", row.get("cancer_history"))

    add_line(lesion_lines, "Body region", row.get("region"))
    d1, d2 = row.get("diameter_1"), row.get("diameter_2")
    if not is_missing(d1) and not is_missing(d2):
        lesion_lines.append(f"- Diameter: {float(d1):g} x {float(d2):g} mm")
    elif not is_missing(d1):
        lesion_lines.append(f"- Diameter: {float(d1):g} mm")
    elif not is_missing(d2):
        lesion_lines.append(f"- Diameter: {float(d2):g} mm")

    add_line(lesion_lines, "Itching", row.get("itch"))
    add_line(lesion_lines, "Growing", row.get("grew"))
    add_line(lesion_lines, "Pain", row.get("hurt"))
    add_line(lesion_lines, "Recent change", row.get("changed"))
    add_line(lesion_lines, "Bleeding", row.get("bleed"))
    add_line(lesion_lines, "Elevation", row.get("elevation"))
    add_line(lesion_lines, "Biopsied", row.get("biopsed"))

    # PAD has no benign_malignant column in the current dump, so malignancy is omitted.
    malignancy = resolve_malignancy(row.get("benign_malignant"))
    code = str(row["diagnostic"]).strip().upper()
    name = PAD_DIAGNOSTIC_NAMES.get(code, code.lower())
    prompt = build_clinical_prompt(
        patient_lines=patient_lines,
        lesion_lines=lesion_lines,
        include_malignancy=malignancy is not None,
    )
    label = build_assistant_label(diagnosis=name, code=code, malignancy=malignancy)
    return prompt, label


def isic18_prompt_and_label(row: Mapping[str, Any]) -> tuple[str, str]:
    """ISIC 2018 Task 3: one-hot class + optional ``diagnosis_confirm_type`` (train only)."""
    patient_lines: list[str] = []
    lesion_lines: list[str] = []

    age = row.get("age_approx")
    if not is_missing(age):
        patient_lines.append(f"- Age: {int(float(age))}")
    if not is_missing(row.get("sex")):
        patient_lines.append(f"- Gender: {str(row['sex']).strip().lower()}")

    add_line(lesion_lines, "Body region", row.get("anatom_site_general"))
    add_line(lesion_lines, "Image type", row.get("image_type"))
    add_line(lesion_lines, "Melanocytic", row.get("melanocytic"))
    add_line(lesion_lines, "Biopsied", row.get("concomitant_biopsy"))
    add_line(lesion_lines, "Diagnosis confirmation", row.get("diagnosis_confirm_type"))

    name, code, malignancy = _closed_class_from_row(
        row,
        name_lookup=ISIC18_DIAGNOSTIC_CODES,
        default_malignancy=row.get("malignancy"),
    )
    prompt = build_clinical_prompt(
        patient_lines=patient_lines,
        lesion_lines=lesion_lines,
        include_malignancy=malignancy is not None,
    )
    label = build_assistant_label(diagnosis=name, code=code, malignancy=malignancy)
    return prompt, label


def ddi_prompt_and_label(row: Mapping[str, Any]) -> tuple[str, str]:
    """DDI: ``DDI_file``, ``disease``, ``malignant``; ignore ``disease-type``."""
    patient_lines: list[str] = []
    if not is_missing(row.get("skin_tone")):
        try:
            key = int(float(row["skin_tone"]))  # type: ignore[arg-type]
            fitz = FITZPATRICK_GROUPS.get(key, str(key))
            patient_lines.append(f"- Fitzpatrick skin type: {fitz}")
        except (TypeError, ValueError):
            pass

    malignancy = resolve_malignancy(row.get("malignant"))
    if malignancy is None:
        malignancy = resolve_malignancy(row.get("benign_malignant"))

    disease = str(row["disease"]).strip().lower()
    name = pretty_disease_name(disease)
    code = DDI_DIAGNOSTIC_CODES.get(disease, disease.upper().replace("-", "_")[:8])
    prompt = build_clinical_prompt(
        patient_lines=patient_lines,
        lesion_lines=None,
        include_malignancy=malignancy is not None,
    )
    label = build_assistant_label(diagnosis=name, code=code, malignancy=malignancy)
    return prompt, label


def pad_image_id(row: Mapping[str, Any]) -> str:
    return str(row["img_id"])


def isic18_image_id(row: Mapping[str, Any]) -> str:
    raw = row.get("img_id", row.get("isic_id", row.get("image")))
    stem = str(raw).strip()
    if stem.lower().endswith((".jpg", ".jpeg", ".png")):
        return stem
    return f"{stem}.jpg"


def ddi_image_id(row: Mapping[str, Any]) -> str:
    raw = row.get("img_id", row.get("DDI_file"))
    return str(raw).strip()


def isic18_image_rel(row: Mapping[str, Any]) -> str:
    if not is_missing(row.get("image_rel")):
        return str(row["image_rel"]).strip()
    return image_rel_path(isic18_image_id(row))


def milk10k_image_rel(row: Mapping[str, Any]) -> str:
    if not is_missing(row.get("image_rel")):
        return str(row["image_rel"]).strip()
    lesion_id = str(row["lesion_id"]).strip()
    isic_id = str(row["isic_id"]).strip()
    return f"MILK10k_Training_Input/{lesion_id}/{isic_id}.jpg"


def hc_image_rel(row: Mapping[str, Any]) -> str:
    if not is_missing(row.get("image_rel")):
        return str(row["image_rel"]).strip()
    return image_rel_path(str(row["imageCropped"]).strip())


def derm1m_image_rel(row: Mapping[str, Any]) -> str:
    if not is_missing(row.get("image_rel")):
        return str(row["image_rel"]).strip()
    return f"images/{str(row['filename']).strip()}"


def _closed_class_from_row(
    row: Mapping[str, Any],
    *,
    name_lookup: Mapping[str, str],
    default_malignancy: Any = None,
) -> tuple[str, str, str | None]:
    malignancy = resolve_malignancy(default_malignancy)
    if malignancy is None:
        malignancy = resolve_malignancy(row.get("benign_malignant"))
    if not is_missing(row.get("diagnosis")) and not is_missing(row.get("code")):
        return str(row["diagnosis"]).strip().lower(), str(row["code"]).strip().upper(), malignancy
    name = str(row.get("diagnosis", "")).strip().lower()
    code = name_lookup.get(name, name.upper().replace(" ", "_")[:8])
    return name, code, malignancy


def _normalize_image_type(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip().lower()
    if text.startswith("clinical"):
        return "clinical"
    if "dermo" in text:
        return "dermoscopic"
    return text.replace("_", " ")


def _fitzpatrick_from_class(value: Any) -> str | None:
    if is_missing(value):
        return None
    try:
        level = int(float(value))
    except (TypeError, ValueError):
        return None
    return FITZPATRICK_ROMAN.get(level)


def milk10k_prompt_and_label(row: Mapping[str, Any]) -> tuple[str, str]:
    """MILK10k: per-image metadata + lesion-level one-hot class."""
    patient_lines: list[str] = []
    lesion_lines: list[str] = []

    age = row.get("age_approx")
    if not is_missing(age):
        patient_lines.append(f"- Age: {int(float(age))}")
    if not is_missing(row.get("sex")):
        patient_lines.append(f"- Gender: {str(row['sex']).strip().lower()}")
    fitz = _fitzpatrick_from_class(row.get("skin_tone_class"))
    if fitz:
        patient_lines.append(f"- Fitzpatrick skin type: {fitz}")

    if not is_missing(row.get("site")):
        lesion_lines.append(f"- Body region: {str(row['site']).strip().replace('_', '/')}")
    image_type = _normalize_image_type(row.get("image_type"))
    if image_type:
        lesion_lines.append(f"- Image type: {image_type}")
    add_line(lesion_lines, "Diagnosis confirmation", row.get("diagnosis_confirm_type"))

    name, code, malignancy = _closed_class_from_row(
        row,
        name_lookup=DIAGNOSTIC_CODES,
        default_malignancy=row.get("malignancy"),
    )
    prompt = build_clinical_prompt(
        patient_lines=patient_lines,
        lesion_lines=lesion_lines,
        include_malignancy=malignancy is not None,
    )
    label = build_assistant_label(diagnosis=name, code=code, malignancy=malignancy)
    return prompt, label


_HC_EXACT = {
    "bcc": "BCC",
    "scc": "SCC",
    "cec": "SCC",
    "mel": "MEL",
    "nv": "NEV",
    "nid": "NEV",
    "akiec": "ACK",
    "bkl": "BKL",
    "df": "DF",
    "vasc": "VASC",
    "dermatofibroma": "DF",
    "melanose solar": "BKL",
    "melanose sola": "BKL",
    "melanosae solar": "BKL",
    "melanise solar": "BKL",
    "menalose solar": "BKL",
    "melanocito solar": "BKL",
    "melanose": "BKL",
    "lentigo solar": "BKL",
    "lentigo": "BKL",
    "lentigo simples": "BKL",
    "dermatose solar": "BKL",
    "fotodano solar": "BKL",
    "queratose seborreica": "BKL",
    "queratose seborreia": "BKL",
    "acrocordon": "OTHER",
    "acrocodon": "OTHER",
    "fibroma mole": "OTHER",
    "fibroma": "OTHER",
}


def _hc_normalize_diagnosis(text: str) -> str:
    text = str(text).strip().lower()
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _hc_rule_code(norm: str) -> str | None:
    if any(
        x in norm
        for x in (
            "a esclarecer",
            "nao sera feit",
            "area de biopsia",
            "cicatriz de cbc",
            "cicatriz de erisipela",
            "complementar o diagnostico",
        )
    ):
        return "OTHER"
    if "melanoma" in norm or "lentigo maligno" in norm:
        return "MEL"
    if any(
        x in norm
        for x in (
            "espinocelular",
            "doenca de bowen",
            "bowen",
            "ceratoacantoma",
            "ceratoacantose",
            "cec in situ",
            "cec bowen",
            "cec doenca",
            "onicopapiloma ou cec",
        )
    ):
        return "SCC"
    if re.search(r"\bcec\b", norm) and "cbc" not in norm:
        return "SCC"
    if "basocelular" in norm or re.search(r"\bbcc\b", norm) or re.search(r"\bcbc\b", norm):
        if "cec" in norm and "basocelular" not in norm and "bcc" not in norm:
            return "OTHER"
        return "BCC"
    if any(x in norm for x in ("akiec", "queratose actinica", "ceratose actinica", "queilite actinica")):
        return "ACK"
    if any(
        x in norm
        for x in (
            "nevo rubi",
            "nevus rubi",
            "angioma",
            "hemangioma",
            "angioqueratoma",
            "granuloma piogenico",
        )
    ) or re.search(r"\bvasc\b", norm):
        return "VASC"
    if any(
        x in norm
        for x in (
            "queratose seborre",
            "dermatose papulosa nigr",
            "dermatose populosa nigr",
            "melanose solar",
            "melanose sola",
            "lentigo solar",
            "bkl",
        )
    ):
        return "BKL"
    if any(
        x in norm
        for x in (
            "nevo ",
            "nevus ",
            "nevo melanocit",
            "nevus melanocit",
            "nevo intraderm",
            "nevus intraderm",
            "nevo juncional",
            "nevo displas",
            "nevus displas",
            "nevo halo",
            "nevus halo",
            "nevo azul",
            "nevus azul",
            "nevo azulado",
            "nevo sebaceo",
            "nevus sebaceo",
            "nevo spilus",
            "nevus spilus",
        )
    ) or norm in {"nevo", "nevus", "nv", "nid"}:
        return "NEV"
    if re.search(r"\bnevo\b", norm) or re.search(r"\bnevus\b", norm):
        return "NEV"
    if "dermatofibroma" in norm or re.search(r"\bdf\b", norm):
        return "DF"
    return None


def hc_pick_raw_diagnosis(row: Mapping[str, Any]) -> str | None:
    for key in ("examResult", "clinicDiagnosis"):
        value = row.get(key)
        if is_missing(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def hc_map_diagnosis(raw: str | None) -> tuple[str, str]:
    """Map to ISIC-like when possible; otherwise keep free-text diagnosis + slug code."""
    if raw is None or not str(raw).strip():
        return "unknown", "UNK"
    text = str(raw).strip()
    norm = _hc_normalize_diagnosis(text)
    code = _HC_EXACT.get(norm) or _hc_rule_code(norm)
    if code is not None and code != "OTHER" and code in HC_CLASS_NAMES:
        return HC_CLASS_NAMES[code], code
    pretty = re.sub(r"\s+", " ", re.sub(r"[_/]+", " ", text)).strip().lower()
    token = re.sub(r"[^A-Za-z0-9]", "", text)
    if re.fullmatch(r"[A-Za-z]{2,8}", token):
        slug = token.upper()
    else:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", text.upper()).strip("_")[:12] or "UNK"
    return pretty, slug


def hc_prompt_and_label(row: Mapping[str, Any]) -> tuple[str, str]:
    """HC: cropped clinical photo + phototype / body part / free-text description."""
    patient_lines: list[str] = []
    lesion_lines: list[str] = []
    if not is_missing(row.get("phototype")):
        patient_lines.append(f"- Fitzpatrick skin type: {str(row['phototype']).strip()}")
    if not is_missing(row.get("bodyPart")):
        lesion_lines.append(f"- Body region: {str(row['bodyPart']).strip()}")
    if not is_missing(row.get("description")):
        lesion_lines.append(f"- Clinical description: {str(row['description']).strip()}")

    name, code = hc_map_diagnosis(hc_pick_raw_diagnosis(row))
    malignancy = resolve_malignancy(row.get("benign_malignant"))
    prompt = build_clinical_prompt(
        patient_lines=patient_lines,
        lesion_lines=lesion_lines,
        include_malignancy=malignancy is not None,
    )
    label = build_assistant_label(diagnosis=name, code=code, malignancy=malignancy)
    return prompt, label


def _pretty_name(value: str) -> str:
    text = str(value).strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _class_code(name: str, *, max_len: int = 32) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _pretty_name(name)).strip("_")
    return (slug.upper()[:max_len] if slug else "UNK")


def derm1m_prompt_and_label(row: Mapping[str, Any]) -> tuple[str, str]:
    """Derm1M labeled pretrain rows; placeholders like ``No age information`` are missing."""
    patient_lines: list[str] = []
    lesion_lines: list[str] = []

    if not derm1m_is_missing(row.get("age")):
        age_raw = str(row["age"]).strip()
        try:
            age_txt = str(int(float(age_raw)))
        except (TypeError, ValueError):
            age_txt = age_raw
        patient_lines.append(f"- Age: {age_txt}")
    if not derm1m_is_missing(row.get("gender")):
        patient_lines.append(f"- Gender: {str(row['gender']).strip().lower()}")
    if not derm1m_is_missing(row.get("body_location")):
        lesion_lines.append(f"- Body region: {str(row['body_location']).strip().lower()}")
    if not derm1m_is_missing(row.get("symptoms")):
        lesion_lines.append(f"- Symptoms: {str(row['symptoms']).strip().lower()}")
    if not derm1m_is_missing(row.get("skin_concept")):
        concepts = str(row["skin_concept"]).replace(",", ", ").strip()
        concepts = re.sub(r"\s+", " ", concepts)
        lesion_lines.append(f"- Visual concepts: {concepts.lower()}")

    name = _pretty_name(str(row["disease_label"]))
    code = _class_code(name)
    prompt = build_clinical_prompt(
        patient_lines=patient_lines,
        lesion_lines=lesion_lines,
        include_malignancy=False,
    )
    label = build_assistant_label(diagnosis=name, code=code)
    return prompt, label
