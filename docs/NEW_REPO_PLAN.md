# New Repository Plan

This repository has grown beyond a plain fork of FLoRA. The cleanest path is to keep this checkout as the full working archive, then create a focused public/project repo around the parts that are now yours.

## Suggested Scope

Working title:

- `federated-adapter-composition`
- `nonlinear-flora`
- `layercraft-federated-llm`

Core project narrative:

> Experiments and tooling for federated instruction tuning with heterogeneous, cumulative, and nonlinear low-rank adapters, building on FLoRA-style aggregation and extending it with LayerCraft adapter variants and controlled dataset partitioning.

## What To Move

Source code to keep:

- `main.py` if you still want the original PEFT/FLoRA baseline in the new repo.
- `main_layercraft.py` for LayerCraft adapter verification and experiments.
- `main_layercraft_uniform.py` if it is still an active comparison.
- `main_nonlinear_flora.py` for nonlinear residual adapter experiments.
- `main_linear_flora_cumulative.py` for cumulative linear residual experiments.
- `main_ffa.py` if FFA remains part of the comparison story.
- `client_data_allocation.py` for Dolly/Wizard split generation.
- `client_data_allocation_cola.py` only if CoLA remains in scope.
- `fed_utils/` after removing caches and any unused scratch files.
- `utils/` after removing caches and tightening docs.
- `templates/` if prompt templates are still required.
- `requirements.txt`, ideally simplified into a smaller curated dependency file.

Experiment launchers to keep, but consider moving under `scripts/`:

- `run_experiment1_expressivity.sh`
- `run_experiment2_multiround.sh`
- `run_epoch_round_tuning.sh`
- `run_layercraft_verify.sh`
- `run_all_seeds.sh`
- `run_all_seeds_ffa.sh`
- `run_flora_*.sh`
- `run_wiz.sh`

Analysis artifacts to keep selectively:

- Keep notebooks only if they are cleaned and reproducible.
- Keep final figures only if they support the README or paper-style results.
- Move figures to `assets/` or `results/figures/`.
- Move notebooks to `notebooks/`.

Data to exclude from Git by default:

- `data_wiz/`
- `data_wiz_stratified/`
- `data_dolly/`
- `data_dolly_stratified/`
- `mmlu_test_*.jsonl`
- generated model outputs, checkpoints, logs, and Slurm output

Files to leave behind or remove from the new repo:

- `__pycache__/`
- ad hoc scratch files such as `fed_utils/hehe`
- large raw/generated datasets
- generated model directories
- any notebooks that are exploratory but not meant for readers

## Suggested New Layout

```text
.
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- src/
|   |-- fed_utils/
|   |-- utils/
|-- scripts/
|   |-- run_experiment1_expressivity.sh
|   |-- run_experiment2_multiround.sh
|   |-- run_layercraft_verify.sh
|-- configs/
|   |-- templates/
|-- notebooks/
|-- assets/
|   |-- figures/
|-- docs/
|   |-- data.md
|   |-- experiments.md
```

If you want minimal churn, skip `src/` for now and keep `fed_utils/` and `utils/` at the root. The important part is separating source, scripts, data, notebooks, and figures.

For a cleaner public repo that avoids copying the unlicensed upstream implementation, prefer the rewrite plan in `REWRITE_EXTRACTION_PLAN.md`.

## README Story

The README should lead with your contribution, not the upstream fork.

Recommended sections:

1. Project name and one-sentence purpose.
2. What is new compared with FLoRA.
3. Method overview: baseline FLoRA, cumulative linear adapters, nonlinear adapters, LayerCraft adapter swapping.
4. Repository layout.
5. Installation.
6. Data preparation.
7. Quickstart commands.
8. Experiment recipes.
9. Results and figures.
10. Attribution to FLoRA and LayerCraft.
11. License and citation.

## Attribution And License Checklist

- Keep clear attribution to the original FLoRA paper and repository.
- Check the upstream repository license before choosing your new repo license.
- If upstream has no license, do not assume you can relicense its code as open source without permission.
- Clearly mark which parts are inherited, modified, and newly added.
- Add citations for FLoRA, PEFT, Hugging Face Transformers, and LayerCraft if used.

## First Migration Commands

These commands assume you create the new repo beside this one.

```bash
mkdir ../federated-adapter-composition
cd ../federated-adapter-composition
git init
```

Then copy source files intentionally rather than copying the whole fork.

Recommended first commit:

```text
Initial focused project extraction

- Add federated adapter training entrypoints
- Add adapter aggregation and client utilities
- Add dataset split generation utilities
- Add experiment scripts and README draft
```

## Suggested `.gitignore`

```gitignore
__pycache__/
*.py[cod]
.ipynb_checkpoints/

data_*/
mmlu_test_*.jsonl
*.jsonl

logs/
*.out
*.err

*/checkpoint-*/
checkpoint-*/
outputs/
results/raw/

.env
.venv/
venv/
```
