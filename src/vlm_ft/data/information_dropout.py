from __future__ import annotations

import copy
import random
import re
from typing import Any

from torch.utils.data import Dataset

_BULLET_RE = re.compile(r"^- [^:]+:\s*.+")
_SECTION_HEADERS = {"Patient context:", "Lesion context:"}
_MINIMAL_PROMPT = (
    "Analyze this skin lesion image.\n"
    "\n"
    'Respond with JSON only using this schema: '
    '{"diagnosis": "<name>", "code": "<CODE>"}.'
)


def apply_information_dropout(text: str, rate: float, rng: random.Random) -> str:
    """Keep each originally observed metadata bullet with probability ``rate``.

    ``rate=1`` is identity (observed context). ``rate=0`` (or all fields
    dropped) yields a minimal image-only diagnosis prompt. Intermediate rates
    implement the stochastic-context training regime.
    """
    if rate >= 1.0:
        return text

    lines = text.splitlines()
    intro: list[str] = []
    sections: list[tuple[str, list[str]]] = []  # (header, kept bullets)
    question: list[str] = []
    current_header: str | None = None
    current_bullets: list[str] = []
    past_intro = False

    def _flush_section() -> None:
        nonlocal current_header, current_bullets
        if current_header is not None and current_bullets:
            sections.append((current_header, current_bullets))
        current_header = None
        current_bullets = []

    for line in lines:
        stripped = line.strip()
        if stripped in _SECTION_HEADERS:
            past_intro = True
            _flush_section()
            current_header = stripped
            current_bullets = []
            continue
        if _BULLET_RE.match(stripped):
            past_intro = True
            if current_header is None:
                # Orphan bullets — treat as a synthetic section without header.
                current_header = ""
            if rate > 0.0 and rng.random() < rate:
                current_bullets.append(stripped)
            continue
        if stripped.startswith("What is the diagnosis") or stripped.startswith("Respond with JSON"):
            _flush_section()
            question.append(stripped)
            continue
        if not past_intro:
            intro.append(line.rstrip())
        # blank / other lines between sections are ignored and rebuilt

    _flush_section()

    if not any(bullets for _, bullets in sections):
        if question:
            return "Analyze this skin lesion image.\n\n" + "\n".join(question)
        return _MINIMAL_PROMPT

    body: list[str] = []
    intro_text = "\n".join(intro).strip()
    if intro_text and "clinical information" in intro_text.lower():
        body.append(intro_text)
    else:
        body.append("Analyze this skin lesion image using the clinical information below.")

    for i, (header, bullets) in enumerate(sections):
        if body:
            body.append("")
        if header:
            body.append(header)
        body.extend(bullets)

    body.append("")
    if question:
        body.extend(question)
    else:
        body.append(
            'Respond with JSON only using this schema: '
            '{"diagnosis": "<name>", "code": "<CODE>"}.'
        )

    return "\n".join(body)


def drop_metadata_from_messages(
    messages: list[dict[str, Any]],
    rate: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Apply metadata masking to the first user text part; leave the assistant intact."""
    if rate >= 1.0:
        return messages

    out = copy.deepcopy(messages)
    for message in out:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = apply_information_dropout(content, rate, rng)
            break
        if not isinstance(content, list):
            break
        for part in content:
            if part.get("type") == "text" and "text" in part:
                part["text"] = apply_information_dropout(part["text"], rate, rng)
                break
        break
    return out


class InformationDropoutDataset(Dataset):
    """On-the-fly metadata masking wrapper for the training split only."""

    def __init__(self, dataset: Dataset, information_rate: float) -> None:
        if not 0.0 <= information_rate <= 1.0:
            raise ValueError(f"information_rate must be in [0, 1], got {information_rate!r}")
        self.dataset = dataset
        self.information_rate = float(information_rate)
        self._rng = random.Random()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.dataset[index]
        if self.information_rate >= 1.0:
            return example
        messages = example.get("messages")
        if messages is None:
            return example
        return {
            **example,
            "messages": drop_metadata_from_messages(messages, self.information_rate, self._rng),
        }
