"""Dataset schema normalization for instruction-tuning records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DOLLY = "dolly"
ALPACA = "alpaca"
INSTRUCTION_OUTPUT = "instruction_output"


@dataclass(frozen=True)
class PromptRecord:
    instruction: str
    context: str | None
    response: str
    schema: str


def detect_schema(record: Mapping[str, object]) -> str:
    """Identify a supported instruction-tuning record shape."""
    keys = set(record)

    if {"instruction", "context", "response"} <= keys:
        return DOLLY
    if {"instruction", "input", "output"} <= keys:
        return ALPACA
    if {"instruction", "output"} <= keys and "response" not in keys:
        return INSTRUCTION_OUTPUT

    expected = (
        "Dolly: instruction/context/response; "
        "Alpaca: instruction/input/output; "
        "instruction-output: instruction/output"
    )
    raise ValueError(f"Unsupported dataset schema. Expected one of: {expected}.")


def normalize_record(record: Mapping[str, object]) -> PromptRecord:
    """Return canonical prompt fields from a supported dataset record."""
    schema = detect_schema(record)

    if schema == DOLLY:
        return PromptRecord(
            instruction=str(record["instruction"]),
            context=_optional_text(record.get("context")),
            response=str(record["response"]),
            schema=schema,
        )

    if schema == ALPACA:
        return PromptRecord(
            instruction=str(record["instruction"]),
            context=_optional_text(record.get("input")),
            response=str(record["output"]),
            schema=schema,
        )

    return PromptRecord(
        instruction=str(record["instruction"]),
        context=None,
        response=str(record["output"]),
        schema=schema,
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None

