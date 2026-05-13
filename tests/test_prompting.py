from fed_adapter.data.prompting import get_template
from fed_adapter.data.schema import normalize_record


def test_alpaca_template_formats_with_input():
    template = get_template("alpaca")
    record = normalize_record({"instruction": "Summarize", "input": "Long note", "output": "Short"})

    prompt = template.format(record)

    assert "### Instruction:" in prompt
    assert "Long note" in prompt
    assert prompt.endswith("Short")


def test_alpaca_template_can_exclude_response():
    template = get_template("alpaca")
    record = normalize_record({"instruction": "Say hi", "output": "Hi"})

    prompt = template.format(record, include_response=False)

    assert "Say hi" in prompt
    assert not prompt.endswith("Hi")

