from __future__ import annotations

import json
import re
from typing import Any

# Codes used by the PAD adapters and the other processed sources in this repo.
DIAGNOSTIC_CODES = frozenset(
    {
        "ACK",
        "ASCNR",
        "BCC",
        "BKL",
        "BN",
        "DF",
        "DYSN",
        "EN",
        "MEL",
        "NEV",
        "NLS",
        "PSCNR",
        "SCC",
        "SEK",
        "SL",
        "VASC",
        "BEN_OTH",
        "INF",
        "MAL_OTH",
    }
)
UNKNOWN_LABEL = "UNKNOWN"

DIAGNOSTIC_NAMES = {
    "BCC": "basal cell carcinoma",
    "SCC": "squamous cell carcinoma",
    "MEL": "melanoma",
    "NEV": "nevus",
    "SEK": "seborrheic keratosis",
    "ACK": "actinic keratosis",
    "BKL": "pigmented benign keratosis",
    "VASC": "vascular lesion",
    "DF": "dermatofibroma",
    "DYSN": "dysplastic nevus",
    "BN": "blue nevus",
    "EN": "epidermal nevus",
    "NLS": "nevus lipomatosus superficialis",
    "ASCNR": "atypical spindle cell nevus of reed",
    "PSCNR": "pigmented spindle cell nevus of reed",
    "SL": "solar lentigo",
    "BEN_OTH": "benign other",
    "INF": "inflammatory or infectious disease",
    "MAL_OTH": "malignant other",
}

_CODE_PATTERN = re.compile(r"\(([A-Z0-9_]{2,12})\)")
_MALIGNANCY_PATTERN = re.compile(r"\b(benign|malignant)\b", re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if part.get("type") == "text"]
            return " ".join(parts)
    return ""


def user_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") == "user"]


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    if not text or not text.strip():
        return None
    raw = text.strip()
    block = _JSON_BLOCK_RE.search(raw)
    if block:
        raw = block.group(1)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Tolerate leading/trailing prose around a JSON object.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def parse_diagnosis(text: str) -> str:
    if not text or not text.strip():
        return UNKNOWN_LABEL

    obj = _extract_json_obj(text)
    if obj is not None:
        code = obj.get("code")
        if isinstance(code, str) and code.strip():
            return code.strip().upper()
        diagnosis = obj.get("diagnosis")
        if isinstance(diagnosis, str) and diagnosis.strip():
            name = diagnosis.strip().lower()
            for known_code, known_name in DIAGNOSTIC_NAMES.items():
                if known_name == name or known_name in name:
                    return known_code
            return diagnosis.strip()
        return UNKNOWN_LABEL

    match = _CODE_PATTERN.search(text.upper())
    if match:
        code = match.group(1)
        if code in DIAGNOSTIC_CODES or len(code) >= 2:
            return code

    lowered = text.lower()
    for code, name in DIAGNOSTIC_NAMES.items():
        if name in lowered:
            return code
        if code.lower() in lowered:
            return code

    return UNKNOWN_LABEL


def parse_malignancy(text: str) -> str | None:
    if not text:
        return None

    obj = _extract_json_obj(text)
    if obj is not None:
        value = obj.get("malignancy")
        if value is None:
            return None
        if isinstance(value, str):
            label = value.strip().lower()
            if label in {"benign", "malignant"}:
                return label
            if label in {"", "null", "none", "unknown"}:
                return None
        return None

    match = _MALIGNANCY_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).lower()


def extract_gold_labels(messages: list[dict[str, Any]]) -> dict[str, str | None]:
    text = _assistant_text(messages)
    return {
        "diagnosis": parse_diagnosis(text),
        "malignancy": parse_malignancy(text),
    }
