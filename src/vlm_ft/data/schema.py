from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ContentPart(BaseModel):
    type: Literal["text", "image"]
    text: str | None = None
    image: str | None = None

    @model_validator(mode="after")
    def check_fields(self) -> ContentPart:
        if self.type == "text":
            if not self.text:
                raise ValueError("text content parts require a non-empty 'text' field")
        elif self.type == "image":
            if not self.image:
                raise ValueError("image content parts require a non-empty 'image' field")
        return self


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: list[ContentPart] | str

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: Any) -> list[dict[str, Any]] | list[ContentPart]:
        if isinstance(value, str):
            return [{"type": "text", "text": value}]
        return value


class Sample(BaseModel):
    messages: list[Message] = Field(min_length=1)

    @model_validator(mode="after")
    def require_user_and_assistant(self) -> Sample:
        roles = {m.role for m in self.messages}
        if "user" not in roles:
            raise ValueError("sample must include at least one user message")
        if "assistant" not in roles:
            raise ValueError("sample must include at least one assistant message")
        return self

    def image_paths(self) -> list[str]:
        paths: list[str] = []
        for message in self.messages:
            if isinstance(message.content, str):
                continue
            for part in message.content:
                if part.type == "image" and part.image:
                    paths.append(part.image)
        return paths

    def to_trl_dict(self) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for message in self.messages:
            if isinstance(message.content, str):
                content: Any = message.content
            else:
                content = [part.model_dump(exclude_none=True) for part in message.content]
            messages.append({"role": message.role, "content": content})
        return {"messages": messages}


class DatasetInfo(BaseModel):
    name: str
    num_samples: int
    splits: list[str] = Field(default_factory=lambda: ["train"])
    image_root: str | None = None
    source: str | None = None
    format_version: str = "1.0"


def validate_sample(obj: dict[str, Any]) -> Sample:
    return Sample.model_validate(obj)


def resolve_image_path(image: str, image_root: str | Path | None = None) -> Path:
    path = Path(image)
    if path.is_absolute() or image_root is None:
        return path
    return Path(image_root) / path
