import pytest

from fed_adapter.data.schema import ALPACA, DOLLY, INSTRUCTION_OUTPUT, detect_schema, normalize_record


def test_normalize_dolly_record():
    record = {"instruction": "Say hi", "context": "", "response": "Hi"}

    normalized = normalize_record(record)

    assert detect_schema(record) == DOLLY
    assert normalized.instruction == "Say hi"
    assert normalized.context is None
    assert normalized.response == "Hi"
    assert normalized.schema == DOLLY


def test_normalize_alpaca_record():
    record = {"instruction": "Summarize", "input": "A long note", "output": "Short"}

    normalized = normalize_record(record)

    assert normalized.context == "A long note"
    assert normalized.response == "Short"
    assert normalized.schema == ALPACA


def test_normalize_instruction_output_record():
    record = {"instruction": "Answer", "output": "42"}

    normalized = normalize_record(record)

    assert normalized.context is None
    assert normalized.response == "42"
    assert normalized.schema == INSTRUCTION_OUTPUT


def test_unknown_schema_raises():
    with pytest.raises(ValueError):
        detect_schema({"prompt": "x", "completion": "y"})

