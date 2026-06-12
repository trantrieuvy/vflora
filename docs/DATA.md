# Data Preparation

V-FLoRA does not commit generated client datasets. The repo keeps dataset acquisition and split generation as reproducible commands.

Run all commands from the repository root unless you pass absolute paths.

## WizardLM Source Split

The WizardLM experiments start from the pre-generated 10-client split provided by the original FederatedLLM project. V-FLoRA then creates a stratified keep-sizes split by combining those client files and redistributing records while preserving the original client sizes.

If you already have a local FederatedLLM clone with `data_wiz`, copy or symlink it directly:

```bash
cp -r /path/to/FederatedLLM/data_wiz ./data_wiz
```

Otherwise, fetch only the Wizard split with Git sparse checkout:

```bash
mkdir -p external

git clone \
  --depth 1 \
  --filter=blob:none \
  --sparse \
  https://github.com/ziyaow1010/FederatedLLM.git \
  external/FederatedLLM

cd external/FederatedLLM
git sparse-checkout set data_wiz/10
cd ../..

cp -r external/FederatedLLM/data_wiz ./data_wiz
```

After this, the repo should contain:

```text
data_wiz/10/local_training_0.json
...
data_wiz/10/local_training_9.json
```

Create the stratified Wizard split:

```bash
python -m fed_adapter.cli.split_data \
  --dataset wizard \
  --mode stratified_keep_sizes \
  --num-clients 10 \
  --source-root data_wiz \
  --output-root data_wiz_stratified
```

The output is:

```text
data_wiz_stratified/10/local_training_0.json
...
data_wiz_stratified/10/local_training_9.json
data_wiz_stratified/10/split_metadata.json
```

## Dolly Raw Dataset

The Dolly workflow starts from the Databricks Dolly 15k dataset, writes it as JSONL, creates a Dirichlet split, then creates a stratified keep-sizes split from that Dirichlet split.

Install the data dependencies:

```bash
pip install datasets pandas numpy
```

Download Dolly and write the JSONL file expected by the splitter:

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("databricks/databricks-dolly-15k", split="train")
ds.to_json("new-databricks-dolly-15k.json", orient="records", lines=True)
PY
```

Create the legacy-style Dolly Dirichlet split:

```bash
python -m fed_adapter.cli.split_data \
  --dataset dolly \
  --mode dirichlet \
  --num-clients 10 \
  --dataset-path new-databricks-dolly-15k.json \
  --output-root data_dolly \
  --alpha 0.5 \
  --seed 42 \
  --test-per-category 10
```

Create the stratified Dolly split while preserving the Dirichlet client sizes:

```bash
python -m fed_adapter.cli.split_data \
  --dataset dolly \
  --mode stratified_keep_sizes \
  --num-clients 10 \
  --source-root data_dolly \
  --output-root data_dolly_stratified \
  --seed 42
```

## GLUE Tasks

GLUE splits are generated directly from the Hugging Face `glue` dataset unless `--source-split-dir` is provided. Supported task names are:

```text
cola mnli mrpc qnli qqp rte sst2 stsb wnli
```

Create a stratified QNLI split:

```bash
python -m fed_adapter.cli.split_data \
  --dataset glue \
  --task-name qnli \
  --mode stratified \
  --num-clients 10 \
  --output-root data_qnli_stratified \
  --seed 0
```

Create an IID MRPC split:

```bash
python -m fed_adapter.cli.split_data \
  --dataset glue \
  --task-name mrpc \
  --mode iid \
  --num-clients 10 \
  --output-root data_mrpc \
  --seed 0
```

The output layout matches the FederatedLLM GLUE runner:

```text
data_qnli_stratified/10/local_training_0.json
...
data_qnli_stratified/10/local_training_9.json
data_qnli_stratified/10/global_val.json
data_qnli_stratified/10/split_metadata.json
```

For MNLI, `global_val.json` is the matched validation split and the splitter also writes:

```text
data_mnli_stratified/10/global_val_mismatched.json
```

To recreate a split from an existing FederatedLLM-style directory while changing client count or mode, pass the existing directory:

```bash
python -m fed_adapter.cli.split_data \
  --dataset glue \
  --task-name qnli \
  --mode stratified \
  --num-clients 3 \
  --source-split-dir data_qnli_stratified/10 \
  --output-root data_qnli_stratified \
  --seed 0
```

For STS-B stratified splitting, continuous scores are assigned to quantile buckets before round-robin allocation. Adjust the bucket count with `--stsb-num-label-buckets` when needed.

## Notes

- `data_wiz/`, `data_wiz_stratified/`, `data_dolly/`, `data_dolly_stratified/`, and `data_<glue_task>*/` are ignored by Git.
- `stratified_keep_sizes` always requires an existing source split.
- For WizardLM, the source split comes from FederatedLLM's pre-generated `data_wiz/10` files.
- For Dolly, the source split can be generated from raw Dolly with `--mode dirichlet`.
- Each generated split writes `split_metadata.json` with the split mode, seed, client sizes, and label/category counts.
