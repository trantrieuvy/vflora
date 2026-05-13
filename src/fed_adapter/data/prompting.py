"""Prompt formatting for instruction-tuning datasets."""

from __future__ import annotations

from dataclasses import dataclass

from fed_adapter.data.schema import PromptRecord


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    prompt_input: str
    prompt_no_input: str
    response_split: str

    def format(self, record: PromptRecord, include_response: bool = True) -> str:
        if record.context:
            prompt = self.prompt_input.format(
                instruction=record.instruction,
                input=record.context,
            )
        else:
            prompt = self.prompt_no_input.format(instruction=record.instruction)
        if include_response:
            prompt += record.response
        return prompt


ALPACA_TEMPLATE = PromptTemplate(
    name="alpaca",
    prompt_input=(
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n"
    ),
    prompt_no_input=(
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    ),
    response_split="### Response:",
)


def get_template(name: str = "alpaca") -> PromptTemplate:
    normalized = (name or "alpaca").lower()
    if normalized == "alpaca":
        return ALPACA_TEMPLATE
    raise ValueError(f"Unknown prompt template: {name!r}")

