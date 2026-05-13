# Rewrite Extraction Plan

The goal is to make the new repository a focused implementation of your project, not a redistributed copy of the unlicensed upstream fork. Treat the current repo as a reference implementation and lab archive, then rewrite the small inherited training core into cleaner project-specific modules.

This is not legal advice, but it is a practical engineering path that reduces copied code and makes authorship much clearer.

## What Looks Inherited

Likely inherited or mostly inherited:

- `main.py`
- `fed_utils/client.py`
- `fed_utils/model_aggregation.py`
- `fed_utils/client_participation_scheduling.py`
- `fed_utils/evaluation.py`
- `utils/prompter.py`
- `utils/callbacks.py`
- `templates/*.json`
- `download.py`, `load.py`, `GlobalModel_generated.py`
- `run_wiz.sh`, `new.sh`
- bundled `data_wiz/` and `mmlu_test_*.jsonl`

Do not copy these into the new repo verbatim unless you get permission or the upstream adds a license.

## What Looks Like Your Project Contribution

Good candidates to carry over after cleanup:

- `main_layercraft.py`
- `main_layercraft_uniform.py`
- `main_nonlinear_flora.py`
- `main_linear_flora_cumulative.py`
- `main_ffa.py`
- `fed_utils/client_layercraft.py`
- `fed_utils/model_aggregation_layercraft.py`
- `fed_utils/evaluation_cola.py`, if CoLA remains in scope
- `client_data_allocation.py`
- `client_data_allocation_cola.py`, if CoLA remains in scope
- `utils/dataset_schema.py`
- `utils/tuning_analysis.py`
- experiment scripts created for your runs
- final figures and cleaned notebooks

Even for these, prefer moving them into a cleaner package layout and editing aggressively so the new repo reads as a coherent project.

## Rewrite Targets

### 1. Training Configuration

Create:

```text
src/fed_adapter/config.py
```

Use dataclasses instead of a giant `fire.Fire(fl_finetune)` function signature:

- `ModelConfig`
- `DataConfig`
- `FederatedConfig`
- `AdapterConfig`
- `TrainingConfig`
- `EvaluationConfig`

This makes experiment scripts shorter and makes README commands easier to understand.

### 2. Federated Client

Rewrite `fed_utils/client.py` and `fed_utils/client_layercraft.py` into:

```text
src/fed_adapter/client.py
```

Design:

- `FederatedClient`
- explicit `prepare_dataset()`
- explicit `build_trainer()`
- explicit `train_one_round()`
- adapter checkpoint save/restore injected through an `AdapterBackend`

Avoid copying the original method names and flow. For example, replace `preprare_local_dataset`, `initiate_local_training`, and `terminate_local_training` with a smaller round-level API.

### 3. Adapter Backends

Create:

```text
src/fed_adapter/adapters/base.py
src/fed_adapter/adapters/peft_backend.py
src/fed_adapter/adapters/layercraft_backend.py
src/fed_adapter/adapters/residual.py
src/fed_adapter/adapters/ffa.py
```

Design:

- `AdapterBackend.state_dict(model)`
- `AdapterBackend.load_state_dict(model, weights)`
- `AdapterBackend.inject(model, adapter_config)`
- `AdapterBackend.merge_if_supported(model)`

This separates the federated loop from PEFT/LayerCraft details.

### 4. Aggregation

Rewrite `fed_utils/model_aggregation.py` and consolidate your LayerCraft aggregation into:

```text
src/fed_adapter/aggregation.py
```

Core functions:

- `client_weights(client_sizes: Mapping[int, int]) -> dict[int, float]`
- `weighted_average(states, weights)`
- `zero_pad_by_rank(states, ranks)`
- `stack_linear_lora(states, weights, ranks)`
- `stack_nonlinear_lora(states, weights, ranks)`
- `aggregate_adapters(states, strategy, weights, ranks)`

This is one of the most important rewrites. The upstream version is a nested flag-heavy function. Your new repo should expose aggregation strategies directly and test them with small tensors.

### 5. Client Selection

Rewrite `client_participation_scheduling.py` into:

```text
src/fed_adapter/selection.py
```

Small clean API:

```python
def select_clients(num_clients: int, fraction: float, seed: int) -> list[int]:
    ...
```

Return a sorted list for deterministic logging and aggregation.

### 6. Prompting And Dataset Schema

Rewrite `utils/prompter.py` and preserve your `dataset_schema.py` idea:

```text
src/fed_adapter/data/prompting.py
src/fed_adapter/data/schema.py
src/fed_adapter/data/splits.py
```

Avoid depending on root-relative `templates/`. Instead, use `importlib.resources` or pass a template path explicitly.

### 7. Evaluation

Rewrite `fed_utils/evaluation.py` into:

```text
src/fed_adapter/evaluation.py
```

Keep it narrow:

- load JSONL validation records
- build prompts through the schema/prompting layer
- normalize multiple-choice answers
- return a structured result object instead of only printing

### 8. Experiment Entrypoints

Replace monolithic `main_*.py` scripts with small CLI entrypoints:

```text
src/fed_adapter/cli/train.py
src/fed_adapter/cli/split_data.py
src/fed_adapter/cli/analyze.py
```

Keep strategy-specific logic in modules, not in separate large scripts. The CLI should choose:

- baseline linear LoRA
- stacked FLoRA
- cumulative linear residual
- cumulative nonlinear residual
- FFA
- LayerCraft adapter family

## Minimal First Version

The first new repo does not need every experiment.

Recommended v0:

```text
src/fed_adapter/
  __init__.py
  config.py
  client.py
  aggregation.py
  selection.py
  evaluation.py
  data/
    __init__.py
    prompting.py
    schema.py
    splits.py
  adapters/
    __init__.py
    base.py
    layercraft_backend.py
    residual.py
  cli/
    __init__.py
    train.py
    split_data.py
scripts/
  run_layercraft_verify.sh
  run_experiment1_expressivity.sh
  run_experiment2_multiround.sh
tests/
  test_aggregation.py
  test_selection.py
  test_dataset_schema.py
```

Then add PEFT baseline compatibility only if you still need it for reproduction.

## Rewrite Order

1. Start with pure functions: `selection.py`, `aggregation.py`, `data/schema.py`.
2. Add tests for tensor aggregation behavior.
3. Move dataset split generation into `data/splits.py`.
4. Create `AdapterBackend` interfaces.
5. Rewrite the federated client around the backend interface.
6. Build one CLI path for nonlinear/cumulative LayerCraft training.
7. Add baseline FLoRA reproduction only as a compatibility mode.
8. Move cleaned scripts and README examples to the new CLI.

## Attribution Wording

Use attribution for ideas and provenance, while avoiding copying unlicensed code:

```text
This project was developed after experiments with the public FederatedLLM/FLoRA
research codebase by Ziyao Wang et al. The new implementation restructures the
training, aggregation, dataset splitting, and adapter-composition code for this
project's experiments. The FLoRA paper and repository are cited for the original
federated heterogeneous LoRA method.
```

If any file is still closely adapted from upstream, mark it at the top:

```python
# This module was rewritten for this project after studying the public
# FederatedLLM/FLoRA implementation. Do not assume upstream licensing.
```

## Tests That Matter Most

Add small CPU-only tests before worrying about full LLM training:

- client selection is deterministic for a seed
- FedAvg weighted average gives expected tensors
- zero-padding preserves values and shapes
- linear stacking weights A but not B
- nonlinear stacking weights B but not A
- heterogeneous ranks stack in the expected order
- prompt schema supports Dolly, Wizard, and CoLA if kept

