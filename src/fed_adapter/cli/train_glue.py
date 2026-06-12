"""Federated RoBERTa training for GLUE tasks."""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np

from fed_adapter.aggregation import (
    aggregate_ffa_b,
    normalized_client_weights,
    stack_linear_lora,
    stack_nonlinear_lora,
    weighted_average,
)
from fed_adapter.adapters.ffa import (
    ffa_B_state_dict,
    init_frozen_A,
    init_zero_B,
    inject_ffa_adapters,
    join_ffa_B_state,
    split_ffa_B_state,
)
from fed_adapter.adapters.flora import join_flora_adapter_state, merge_linear_lora_into_model
from fed_adapter.adapters.residual import (
    accumulate_adapters,
    adapter_state_dict,
    inject_residual_adapters,
    split_adapter_state,
)
from fed_adapter.data.splits import GLUE_TASK_TO_KEYS


SERVER_STATE_FILENAME = "server_state.pt"

GLUE_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "stsb": 1,
    "wnli": 2,
}

GLUE_VARIANT_ALIASES = {
    "flora": "flora",
    "linear_flora_cumulative": "linear_flora_cumulative",
    "linear-cumulative-flora": "linear_flora_cumulative",
    "linear_cumulative_flora": "linear_flora_cumulative",
    "nonlinear_flora": "nonlinear_flora",
    "nonlinear-cumulative-flora": "nonlinear_flora",
    "nonlinear_cumulative_flora": "nonlinear_flora",
    "ffa": "ffa",
    "nonlinear-ffa": "ffa",
    "nonlinear_ffa": "ffa",
}

RESIDUAL_VARIANTS = {"flora", "linear_flora_cumulative", "nonlinear_flora"}
CUMULATIVE_RESIDUAL_VARIANTS = {"linear_flora_cumulative", "nonlinear_flora"}

MODEL_ALIASES = {
    "roberta": "roberta-base",
    "roberta-base": "roberta-base",
}


def train(args) -> None:
    torch, transformers, Dataset, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer = _ml_imports()

    variant = _resolve_glue_variant(args)
    task_name = _resolve_task_name(args)
    if task_name not in GLUE_TASK_TO_KEYS:
        raise ValueError(f"Unknown GLUE task: {task_name}")

    heter = bool(getattr(args, "heterogeneous", False) or _optional_bool(getattr(args, "heter", None)))
    local_ranks = _parse_int_list(args.local_ranks)
    if heter and len(local_ranks) < args.num_clients:
        raise ValueError("--local-ranks must provide at least one rank per client")
    if args.max_rounds_per_invocation < 0:
        raise ValueError("--max-rounds-per-invocation must not be negative")
    if args.retain_adapter_every_n_rounds < 0:
        raise ValueError("--retain-adapter-every-n-rounds must not be negative")

    _seed_everything(args.seed, torch, deterministic=args.use_deterministic_algorithms)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = args.data_root / str(args.num_clients)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing federated split: {data_dir}")

    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
    val_records = _read_records(data_dir / "global_val.json")
    val_mismatched_records = (
        _read_records(data_dir / "global_val_mismatched.json")
        if task_name == "mnli" and (data_dir / "global_val_mismatched.json").exists()
        else None
    )
    glue_metric = _load_glue_metric(task_name)

    model_name = MODEL_ALIASES.get(args.model, args.model)
    target_modules = _parse_targets(args.target_modules or "query,value")
    base_scaling = float(args.alpha) / int(args.rank)
    global_ffa_rank = max(local_ranks[: args.num_clients]) if heter else args.rank

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=GLUE_NUM_LABELS[task_name],
        finetuning_task=task_name,
    )
    raw_model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    for parameter in raw_model.parameters():
        parameter.requires_grad = False
    raw_model.to("cpu")

    A_cumulative = None
    B_cumulative = None
    A_ffa = None
    B_ffa = None
    if variant == "ffa":
        A_ffa = init_frozen_A(raw_model, target_modules, global_ffa_rank, args.seed, args.a_init_std)
        B_ffa = init_zero_B(raw_model, target_modules, global_ffa_rank)

    output_dir = args.output_dir / str(args.num_clients)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / SERVER_STATE_FILENAME
    score_list: list[float] = []
    expected_cumulative_rank = 0
    start_round = 0

    if args.resume_from_latest:
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume without server state: {state_path}")
        server_state = torch.load(state_path, map_location="cpu")
        _validate_resume_state(server_state, args, variant)
        if variant == "flora":
            raw_model.load_state_dict(server_state["raw_model_state"])
        else:
            _load_classifier_state(raw_model, server_state["classifier_state"])
        if variant in CUMULATIVE_RESIDUAL_VARIANTS:
            A_cumulative = server_state.get("A_cumulative")
            B_cumulative = server_state.get("B_cumulative")
        if variant == "ffa":
            B_ffa = server_state.get("B_ffa", B_ffa)
        score_list = [float(value) for value in server_state["score_list"]]
        expected_cumulative_rank = int(server_state["expected_cumulative_rank"])
        start_round = int(server_state["completed_round"]) + 1
        _restore_rng_state(server_state["rng_state"], torch)

    if start_round >= args.rounds:
        print(f"All {args.rounds} communication rounds are already complete.")
        return

    end_round = args.rounds
    if args.max_rounds_per_invocation > 0:
        end_round = min(start_round + args.max_rounds_per_invocation, args.rounds)

    print(
        "Federated RoBERTa GLUE tuning\n"
        f"  variant: {variant}\n"
        f"  task_name: {task_name}\n"
        f"  model: {model_name}\n"
        f"  data_dir: {data_dir}\n"
        f"  rounds: {args.rounds}\n"
        f"  executing: {start_round}..{end_round - 1}\n"
        f"  rank/alpha: {args.rank}/{args.alpha}\n"
        f"  heter: {heter}\n"
    )

    for round_id in range(start_round, end_round):
        retain_round_artifacts = (
            round_id == args.rounds - 1
            or (
                args.retain_adapter_every_n_rounds > 0
                and (round_id + 1) % args.retain_adapter_every_n_rounds == 0
            )
        )
        selected_clients = _select_clients_like_federatedllm(args.num_clients, args.client_fraction, round_id)
        print(f"\n=== Round {round_id} ===")
        print(f"  Selected clients: {selected_clients}")

        client_adapter_states = {}
        client_classifier_states = {}
        client_sizes: dict[int, int] = {}

        for client_id in selected_clients:
            client_rank = local_ranks[client_id] if heter else args.rank
            client_alpha = base_scaling * client_rank
            raw_model.to("cpu")
            client_model = AutoModelForSequenceClassification.from_config(config)
            client_model.load_state_dict(raw_model.state_dict())

            if variant in RESIDUAL_VARIANTS:
                use_cumulative = variant in CUMULATIVE_RESIDUAL_VARIANTS
                client_model, adapter_count = inject_residual_adapters(
                    client_model,
                    target_modules=target_modules,
                    rank=client_rank,
                    alpha=client_alpha,
                    dropout=args.dropout,
                    nonlinear=variant == "nonlinear_flora",
                    A_frozen=A_cumulative if use_cumulative else None,
                    B_frozen=B_cumulative if use_cumulative else None,
                    frozen_scaling=base_scaling,
                )
            else:
                assert A_ffa is not None and B_ffa is not None
                client_model, adapter_count = inject_ffa_adapters(
                    client_model,
                    target_modules=target_modules,
                    A_frozen=A_ffa,
                    B_state=B_ffa,
                    scaling=base_scaling,
                    dropout=args.dropout,
                    activation=args.activation,
                    client_rank=client_rank if heter else None,
                )

            _set_trainable_parameters(client_model, variant)
            client_model.to(device)
            train_dataset = _client_dataset(
                Dataset,
                tokenizer,
                data_dir / f"local_training_{client_id}.json",
                sentence1_key,
                sentence2_key,
                task_name,
                args.cutoff_len,
                args.local_val_size,
                seed=args.seed + client_id,
            )
            trainer = _build_trainer(
                transformers,
                client_model,
                tokenizer,
                train_dataset,
                output_dir / "trainer_saved" / f"local_output_{client_id}",
                args,
            )
            trainable = sum(parameter.numel() for parameter in client_model.parameters() if parameter.requires_grad)
            print(f"  Client_{client_id}: rank={client_rank}, adapters={adapter_count}, trainable={trainable:,}")
            trainer.train()

            if variant == "ffa":
                adapter_state = ffa_B_state_dict(client_model)
                client_adapter_states[client_id] = split_ffa_B_state(adapter_state)
            else:
                adapter_state = adapter_state_dict(client_model)
                client_adapter_states[client_id] = adapter_state
            client_classifier_states[client_id] = _classifier_state_dict(client_model)
            client_sizes[client_id] = len(train_dataset)

            checkpoint_dir = output_dir / str(round_id) / f"local_output_{client_id}"
            if retain_round_artifacts:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "adapter": adapter_state,
                        "classifier": client_classifier_states[client_id],
                        "num_examples": len(train_dataset),
                    },
                    checkpoint_dir / "pytorch_model.bin",
                )
            del trainer, client_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        weights = normalized_client_weights(client_sizes)
        classifier_state = weighted_average(client_classifier_states, weights)
        _load_classifier_state(raw_model, classifier_state)

        round_dir = output_dir / str(round_id)
        round_dir.mkdir(parents=True, exist_ok=True)
        ranks = {client_id: (local_ranks[client_id] if heter else args.rank) for client_id in selected_clients}
        expected_round_rank = sum(ranks.values()) if variant in RESIDUAL_VARIANTS else global_ffa_rank

        if variant in RESIDUAL_VARIANTS:
            if variant == "nonlinear_flora":
                round_state = stack_nonlinear_lora(client_adapter_states, weights, ranks)
            else:
                round_state = stack_linear_lora(client_adapter_states, weights, ranks)
            round_A, round_B = split_adapter_state(round_state)
            round_rank = next(iter(round_A.values())).shape[0]
            if variant == "flora":
                merge_linear_lora_into_model(raw_model, round_A, round_B, base_scaling)
                cumulative_rank = round_rank
                expected_cumulative_rank_for_round = expected_round_rank
                if retain_round_artifacts:
                    torch.save(join_flora_adapter_state(round_A, round_B), round_dir / "adapter_model.bin")
            else:
                A_cumulative, B_cumulative = accumulate_adapters(A_cumulative, B_cumulative, round_A, round_B)
                expected_cumulative_rank += expected_round_rank
                cumulative_rank = next(iter(A_cumulative.values())).shape[0]
                expected_cumulative_rank_for_round = expected_cumulative_rank
                if retain_round_artifacts:
                    torch.save(join_flora_adapter_state(round_A, round_B), round_dir / "adapter_model_delta.bin")
                    torch.save(join_flora_adapter_state(A_cumulative, B_cumulative), round_dir / "adapter_model.bin")
        else:
            assert B_ffa is not None
            B_ffa = aggregate_ffa_b(client_adapter_states, weights, B_ffa)
            round_rank = next(iter(B_ffa.values())).shape[1]
            cumulative_rank = round_rank
            expected_cumulative_rank_for_round = global_ffa_rank
            if retain_round_artifacts:
                torch.save(join_ffa_B_state(B_ffa), round_dir / "adapter_model.bin")

        if round_rank != expected_round_rank or cumulative_rank != expected_cumulative_rank_for_round:
            raise ValueError(
                f"Rank sanity check failed at round {round_id}: "
                f"round_rank={round_rank} expected={expected_round_rank}, "
                f"cumulative_or_global_rank={cumulative_rank} expected={expected_cumulative_rank_for_round}"
            )

        eval_model = _build_eval_model(
            copy.deepcopy(raw_model),
            target_modules,
            args,
            variant,
            base_scaling,
            A_cumulative,
            B_cumulative,
            A_ffa,
            B_ffa,
        )
        eval_model.to(device)
        matched_metrics = _evaluate_records(
            eval_model,
            tokenizer,
            val_records,
            sentence1_key,
            sentence2_key,
            task_name,
            args.cutoff_len,
            args.eval_batch_size,
            device,
            torch,
            glue_metric,
        )
        mismatched_metrics = None
        if task_name == "mnli" and val_mismatched_records is not None:
            mismatched_metrics = _evaluate_records(
                eval_model,
                tokenizer,
                val_mismatched_records,
                sentence1_key,
                sentence2_key,
                task_name,
                args.cutoff_len,
                args.eval_batch_size,
                device,
                torch,
                glue_metric,
            )
        round_metrics = _round_metric_summary(task_name, matched_metrics, mismatched_metrics)
        primary_score = float(round_metrics["primary_score"])
        score_list.append(primary_score)
        print(f"  Acc round {round_id}: {primary_score}")
        print(f"  Primary metric: {round_metrics['primary_metric']}={primary_score}")
        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _write_round_config(
            round_dir,
            round_id,
            args,
            variant,
            task_name,
            model_name,
            selected_clients,
            client_sizes,
            weights,
            ranks,
            round_rank,
            expected_round_rank,
            cumulative_rank,
            expected_cumulative_rank_for_round,
            retain_round_artifacts,
            state_path,
            val_records,
            round_metrics,
        )

        server_state = {
            "version": 1,
            "variant": variant,
            "method": variant,
            "task_name": task_name,
            "num_clients": int(args.num_clients),
            "rounds": int(args.rounds),
            "rank": int(args.rank),
            "alpha": float(args.alpha),
            "heterogeneous": bool(heter),
            "seed": int(args.seed),
            "completed_round": int(round_id),
            "classifier_state": _classifier_state_dict(raw_model),
            "A_cumulative": A_cumulative,
            "B_cumulative": B_cumulative,
            "B_ffa": B_ffa,
            "score_list": score_list,
            "expected_cumulative_rank": int(expected_cumulative_rank),
            "rng_state": _rng_state(torch),
        }
        if variant == "flora":
            server_state["raw_model_state"] = _model_state_dict_cpu(raw_model)
        _write_server_state(state_path, server_state, torch)
        print(f"  Saved resumable server state: {state_path}; adapter_artifacts_retained={retain_round_artifacts}")

    if end_round == args.rounds:
        log_path = _log_path_for_variant(args.output_dir, variant, args.num_clients)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as outfile:
            for score in score_list:
                outfile.write(f"{score}\n")
        _mirror_flora_log(args.output_dir, variant, args.num_clients)
        print(f"Final scores: {score_list}")
        print(f"Log saved to {log_path}")
    else:
        print(f"Segment complete through round {end_round - 1}; resume from {state_path}.")


def _resolve_glue_variant(args) -> str:
    raw = args.method or args.variant or "flora"
    try:
        return GLUE_VARIANT_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(f"Unknown GLUE variant/method: {raw}") from exc


def _resolve_task_name(args) -> str:
    if args.task_name:
        return args.task_name.lower()
    name = args.data_root.name
    if name.startswith("data_"):
        name = name[5:]
    for suffix in ("_stratified", "_dirichlet"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.lower()


def _set_trainable_parameters(model, variant: str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        if name.startswith("classifier."):
            parameter.requires_grad = True
        elif variant in RESIDUAL_VARIANTS and (name.endswith(".A_new") or name.endswith(".B_new")):
            parameter.requires_grad = True
        elif variant == "ffa" and name.endswith(".B"):
            parameter.requires_grad = True


def _classifier_state_dict(model) -> dict[str, object]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("classifier.")
    }


def _load_classifier_state(model, state_dict: Mapping[str, object]) -> None:
    named_parameters = dict(model.named_parameters())
    for name, value in state_dict.items():
        if name not in named_parameters:
            raise KeyError(f"Classifier parameter not found: {name}")
        named_parameters[name].data.copy_(value.to(device=named_parameters[name].device, dtype=named_parameters[name].dtype))


def _model_state_dict_cpu(model) -> dict[str, object]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _build_eval_model(
    model,
    target_modules: list[str],
    args,
    variant: str,
    scaling: float,
    A_cumulative,
    B_cumulative,
    A_ffa,
    B_ffa,
):
    if variant in CUMULATIVE_RESIDUAL_VARIANTS:
        if A_cumulative is None or B_cumulative is None:
            raise ValueError("Missing cumulative adapters for evaluation")
        model, _ = inject_residual_adapters(
            model,
            target_modules=target_modules,
            rank=args.rank,
            alpha=args.alpha,
            dropout=0.0,
            nonlinear=variant == "nonlinear_flora",
            A_frozen=A_cumulative,
            B_frozen=B_cumulative,
            frozen_scaling=scaling,
        )
    elif variant == "ffa":
        if A_ffa is None or B_ffa is None:
            raise ValueError("Missing FFA adapters for evaluation")
        model, _ = inject_ffa_adapters(
            model,
            target_modules=target_modules,
            A_frozen=A_ffa,
            B_state=B_ffa,
            scaling=scaling,
            dropout=0.0,
            activation=args.activation,
            client_rank=None,
        )
    return model


def _client_dataset(
    Dataset,
    tokenizer,
    path: Path,
    sentence1_key: str,
    sentence2_key: str | None,
    task_name: str,
    max_seq_length: int,
    local_val_size: float,
    seed: int,
):
    records = _read_records(path)
    dataset = _records_to_dataset(Dataset, records, sentence1_key, sentence2_key, task_name)
    if local_val_size > 0:
        split = dataset.train_test_split(test_size=local_val_size, seed=seed)
        dataset = split["train"]
    dataset = dataset.shuffle(seed=seed)
    return _tokenize_dataset(dataset, tokenizer, sentence1_key, sentence2_key, max_seq_length)


def _records_to_dataset(Dataset, records: list[dict], sentence1_key: str, sentence2_key: str | None, task_name: str):
    labels = [float(record["label"]) if task_name == "stsb" else int(record["label"]) for record in records]
    payload = {
        sentence1_key: [record[sentence1_key] for record in records],
        "label": labels,
    }
    if sentence2_key is not None:
        payload[sentence2_key] = [record[sentence2_key] for record in records]
    return Dataset.from_dict(payload)


def _tokenize_dataset(dataset, tokenizer, sentence1_key: str, sentence2_key: str | None, max_seq_length: int):
    def tokenize_fn(examples):
        if sentence2_key is None:
            tokenized = tokenizer(
                examples[sentence1_key],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
            )
        else:
            tokenized = tokenizer(
                examples[sentence1_key],
                examples[sentence2_key],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
            )
        tokenized["labels"] = examples["label"]
        return tokenized

    remove_columns = [sentence1_key, "label"]
    if sentence2_key is not None:
        remove_columns.append(sentence2_key)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)
    tokenized.set_format("torch")
    return tokenized


def _build_trainer(transformers, model, tokenizer, train_dataset, output_dir: Path, args):
    gradient_accumulation_steps = max(1, args.local_batch_size // args.micro_batch_size)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)
    fp16 = bool(getattr(args, "fp16", False))
    bf16 = bool(getattr(args, "bf16", False))
    train_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=fp16,
        bf16=bf16,
        logging_steps=50,
        evaluation_strategy="no",
        save_strategy="no",
        optim="adamw_torch",
        report_to="none",
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        seed=args.seed,
    )
    return transformers.Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=transformers.default_data_collator,
        tokenizer=tokenizer,
    )


def _evaluate_records(
    model,
    tokenizer,
    records: list[dict],
    sentence1_key: str,
    sentence2_key: str | None,
    task_name: str,
    max_seq_length: int,
    batch_size: int,
    device,
    torch,
    glue_metric,
) -> dict[str, float]:
    model.eval()
    labels = np.array([float(record["label"]) if task_name == "stsb" else int(record["label"]) for record in records])
    predictions = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            if sentence2_key is None:
                encoded = tokenizer(
                    [record[sentence1_key] for record in batch],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
            else:
                encoded = tokenizer(
                    [record[sentence1_key] for record in batch],
                    [record[sentence2_key] for record in batch],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().cpu().numpy()
            if task_name == "stsb":
                predictions.extend(np.squeeze(logits).tolist())
            else:
                predictions.extend(np.argmax(logits, axis=1).tolist())
    model.train()
    return compute_glue_metrics(task_name, np.array(predictions), labels, metric=glue_metric)


def compute_glue_metrics(
    task_name: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    metric=None,
) -> dict[str, float]:
    """Compute GLUE metrics through Hugging Face datasets, matching LoRA-T."""
    task_name = task_name.lower()
    if task_name == "stsb":
        formatted_predictions = predictions.astype(float)
        formatted_labels = labels.astype(float)
    else:
        formatted_predictions = predictions.astype(int)
        formatted_labels = labels.astype(int)

    metric = metric or _load_glue_metric(task_name)
    result = metric.compute(predictions=formatted_predictions, references=formatted_labels)
    result = {key: float(value) for key, value in result.items()}
    if len(result) > 1 and "combined_score" not in result:
        result["combined_score"] = float(np.mean(list(result.values())).item())
    return result


def _round_metric_summary(
    task_name: str,
    matched_metrics: dict[str, float],
    mismatched_metrics: dict[str, float] | None,
) -> dict[str, object]:
    if task_name == "mnli":
        mismatched = mismatched_metrics or {"accuracy": matched_metrics["accuracy"]}
        overall_accuracy = (matched_metrics["accuracy"] + mismatched["accuracy"]) / 2.0
        return {
            "primary_metric": "accuracy",
            "primary_score": overall_accuracy,
            "accuracy": overall_accuracy,
            "mnli_matched": matched_metrics,
            "mnli_mismatched": mismatched,
        }
    if task_name == "cola":
        primary = matched_metrics["matthews_correlation"]
        return {"primary_metric": "matthews_correlation", "primary_score": primary, **matched_metrics}
    if task_name == "stsb":
        primary = matched_metrics["pearson"]
        return {"primary_metric": "pearson", "primary_score": primary, **matched_metrics}
    return {"primary_metric": "accuracy", "primary_score": matched_metrics["accuracy"], **matched_metrics}


def _write_round_config(
    round_dir: Path,
    round_id: int,
    args,
    variant: str,
    task_name: str,
    model_name: str,
    selected_clients: list[int],
    client_sizes: Mapping[int, int],
    weights: Mapping[int, float],
    ranks: Mapping[int, int],
    round_rank: int,
    expected_round_rank: int,
    cumulative_rank: int,
    expected_cumulative_rank: int,
    retain_round_artifacts: bool,
    state_path: Path,
    val_records: list[dict],
    round_metrics: dict[str, object],
) -> None:
    metadata = {
        "round": int(round_id),
        "epoch": int(round_id),
        "variant": variant,
        "method": variant,
        "task_name": task_name,
        "global_model": model_name,
        "rank": int(args.rank),
        "lora_r": int(args.rank),
        "alpha": float(args.alpha),
        "lora_alpha": float(args.alpha),
        "effective_scaling": float(args.alpha) / int(args.rank),
        "heterogeneous": bool(getattr(args, "heterogeneous", False) or _optional_bool(getattr(args, "heter", None))),
        "heter": bool(getattr(args, "heterogeneous", False) or _optional_bool(getattr(args, "heter", None))),
        "local_ranks": [int(rank) for rank in _parse_int_list(args.local_ranks)[: args.num_clients]]
        if (getattr(args, "heterogeneous", False) or _optional_bool(getattr(args, "heter", None)))
        else None,
        "selected_clients": [int(client_id) for client_id in selected_clients],
        "local_dataset_sizes": {str(client_id): int(size) for client_id, size in client_sizes.items()},
        "client_weights": {str(client_id): float(weight) for client_id, weight in weights.items()},
        "client_ranks": {str(client_id): int(rank) for client_id, rank in ranks.items()},
        "local_val_set_size": args.local_val_size,
        "adapter_artifacts_retained": bool(retain_round_artifacts),
        "server_state_path": str(state_path),
        "round_stacked_r": int(round_rank),
        "expected_round_stacked_r": int(expected_round_rank),
        "cumulative_or_global_r": int(cumulative_rank),
        "expected_cumulative_or_global_r": int(expected_cumulative_rank),
        "rank_semantics": _rank_semantics(variant),
        "validation_label_counts": dict(Counter(record["label"] for record in val_records)),
        **round_metrics,
    }
    with (round_dir / "round_config.json").open("w") as outfile:
        json.dump(metadata, outfile, indent=2)


def _rank_semantics(variant: str) -> str:
    return {
        "flora": "merged_linear_residual",
        "linear_flora_cumulative": "cumulative_linear_residual",
        "nonlinear_flora": "cumulative_nonlinear_residual",
        "ffa": "ffa_global_B",
    }[variant]


def _validate_resume_state(server_state: dict, args, variant: str) -> None:
    expected = {
        "variant": variant,
        "task_name": _resolve_task_name(args),
        "num_clients": int(args.num_clients),
        "rounds": int(args.rounds),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
    }
    for name, expected_value in expected.items():
        found = server_state.get(name, server_state.get("method") if name == "variant" else None)
        if found != expected_value:
            raise ValueError(f"Resume state mismatch for {name}: found {found!r}, expected {expected_value!r}")


def _log_path_for_variant(output_dir: Path, variant: str, num_clients: int) -> Path:
    client_dir = output_dir / str(num_clients)
    if variant == "flora":
        return Path(f"{client_dir}log.txt")
    return client_dir / "log.txt"


def _mirror_flora_log(output_dir: Path, variant: str, num_clients: int) -> None:
    if variant != "flora":
        return
    legacy_path = _log_path_for_variant(output_dir, variant, num_clients)
    standard_path = output_dir / str(num_clients) / "log.txt"
    if legacy_path.exists():
        standard_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy_path, standard_path)


def _select_clients_like_federatedllm(num_clients: int, fraction: float, round_id: int) -> list[int]:
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")
    if not 0 < fraction <= 1:
        raise ValueError("client fraction must be in the interval (0, 1]")
    rng_state = np.random.get_state()
    try:
        np.random.seed(round_id)
        count = max(int(fraction * num_clients), 1)
        selected = np.random.choice(np.arange(num_clients), count, replace=False)
        return sorted(int(value) for value in selected)
    finally:
        np.random.set_state(rng_state)


def _read_records(path: Path) -> list[dict]:
    with path.open() as infile:
        return json.load(infile)


def _parse_targets(value: str) -> list[str]:
    targets = [item.strip() for item in value.split(",") if item.strip()]
    if not targets:
        raise ValueError("target modules cannot be empty")
    return targets


def _parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _optional_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _rng_state(torch) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict, torch) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _write_server_state(path: Path, state: dict, torch) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def _seed_everything(seed: int, torch, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def _load_glue_metric(task_name: str):
    try:
        from datasets import load_metric
    except ImportError as exc:
        raise RuntimeError("GLUE metric computation requires datasets.") from exc

    try:
        return load_metric("glue", task_name, trust_remote_code=True)
    except TypeError:
        return load_metric("glue", task_name)


def _ml_imports():
    try:
        import torch
        import transformers
        from datasets import Dataset
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "GLUE training requires torch, transformers, datasets, and numpy. "
            "Install the training requirements before running this command."
        ) from exc
    return torch, transformers, Dataset, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
