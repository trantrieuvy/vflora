"""Train federated LoRA variant experiments."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from pathlib import Path

from fed_adapter.data.prompting import get_template
from fed_adapter.data.schema import normalize_record
from fed_adapter.selection import select_clients


MODEL_ALIASES = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama-7b": "huggyllama/llama-7b",
    "gpt2": "gpt2",
}

VARIANT_ALIASES = {
    "linear-cumulative-flora": "linear-cumulative-flora",
    "cumulative-linear": "linear-cumulative-flora",
    "nonlinear-cumulative-flora": "nonlinear-cumulative-flora",
    "nonlinear": "nonlinear-cumulative-flora",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V-FLoRA federated adapter training.")
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_ALIASES),
        default="nonlinear-cumulative-flora",
        help=(
            "Federated adapter method. Short aliases 'nonlinear' and "
            "'cumulative-linear' are kept for compatibility."
        ),
    )
    parser.add_argument("--model", default="tinyllama")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--local-ranks", default="64,32,16,16,8,8,4,4,4,4")
    parser.add_argument("--local-batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--local-val-size", type=int, default=0)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--prompt-template", default="alpaca")
    parser.add_argument("--train-on-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--keep-local-checkpoints", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)


def train(args: argparse.Namespace) -> None:
    torch, transformers, load_dataset, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, tqdm = _ml_imports()
    from fed_adapter.aggregation import normalized_client_weights, stack_linear_lora, stack_nonlinear_lora
    from fed_adapter.adapters.residual import (
        accumulate_adapters,
        adapter_state_dict,
        inject_residual_adapters,
        join_adapter_state,
        split_adapter_state,
    )

    _seed_everything(args.seed, torch)

    data_dir = args.data_root / str(args.num_clients)
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing client data directory: {data_dir}")

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    local_ranks = [int(item) for item in args.local_ranks.split(",") if item.strip()]
    if args.heterogeneous and len(local_ranks) < args.num_clients:
        raise ValueError("--local-ranks must provide at least one rank per client")

    output_dir = args.output_dir / str(args.num_clients)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODEL_ALIASES.get(args.model, args.model)
    dtype = _torch_dtype(args.torch_dtype, torch)
    token = os.environ.get("HF_TOKEN")
    device_map = _device_map()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    tokenizer.padding_side = "left"

    for parameter in model.parameters():
        parameter.requires_grad = False

    template = get_template(args.prompt_template)
    frozen_A = None
    frozen_B = None
    frozen_scaling = args.alpha / args.rank
    accuracies: list[float] = []
    variant = VARIANT_ALIASES[args.variant]
    nonlinear = variant == "nonlinear-cumulative-flora"

    for round_id in tqdm(range(args.rounds), desc="federated rounds"):
        selected = select_clients(args.num_clients, args.client_fraction, seed=round_id)
        client_sizes: dict[int, int] = {}
        client_states = {}

        for client_id in selected:
            rank = local_ranks[client_id] if args.heterogeneous else args.rank
            alpha = frozen_scaling * rank
            model.to("cpu")
            client_model, adapter_count = inject_residual_adapters(
                copy.deepcopy(model),
                target_modules=target_modules,
                rank=rank,
                alpha=alpha,
                dropout=args.dropout,
                nonlinear=nonlinear,
                A_frozen=frozen_A,
                B_frozen=frozen_B,
                frozen_scaling=frozen_scaling,
            )
            client_model.to(args.device)
            train_dataset, eval_dataset = _client_dataset(
                load_dataset,
                data_dir / f"local_training_{client_id}.json",
                tokenizer,
                template,
                args.cutoff_len,
                args.train_on_inputs,
                args.local_val_size,
                seed=args.seed,
            )
            trainer = _build_trainer(
                transformers,
                client_model,
                tokenizer,
                train_dataset,
                eval_dataset,
                output_dir / "trainer_saved" / f"local_output_{client_id}",
                args,
            )
            print(f"round={round_id} client={client_id} rank={rank} adapters={adapter_count}")
            trainer.train()
            state = adapter_state_dict(client_model)
            checkpoint_dir = output_dir / str(round_id) / f"local_output_{client_id}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, checkpoint_dir / "adapter_model.bin")
            client_sizes[client_id] = len(train_dataset)
            client_states[client_id] = state
            del trainer, client_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        weights = normalized_client_weights(client_sizes)
        ranks = {client_id: (local_ranks[client_id] if args.heterogeneous else args.rank) for client_id in selected}
        if nonlinear:
            round_state = stack_nonlinear_lora(client_states, weights, ranks)
        else:
            round_state = stack_linear_lora(client_states, weights, ranks)

        round_A, round_B = split_adapter_state(round_state)
        frozen_A, frozen_B = accumulate_adapters(frozen_A, frozen_B, round_A, round_B)
        cumulative_state = join_adapter_state(frozen_A, frozen_B)

        round_dir = output_dir / str(round_id)
        torch.save(round_state, round_dir / "adapter_model_delta.bin")
        torch.save(cumulative_state, round_dir / "adapter_model.bin")
        _write_round_metadata(round_dir, round_id, variant, selected, ranks, weights, round_A, round_B, frozen_A, frozen_B, args)

        if args.eval_path:
            eval_model, _ = inject_residual_adapters(
                copy.deepcopy(model),
                target_modules=target_modules,
                rank=args.rank,
                alpha=args.alpha,
                dropout=0.0,
                nonlinear=nonlinear,
                A_frozen=frozen_A,
                B_frozen=frozen_B,
                frozen_scaling=frozen_scaling,
            )
            eval_model.to(args.device)
            accuracy = _evaluate_mmlu(eval_model, tokenizer, template, args.eval_path, args.device, GenerationConfig, torch, tqdm)
            accuracies.append(accuracy)
            del eval_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not args.keep_local_checkpoints:
            for path in round_dir.glob("local_output_*"):
                shutil.rmtree(path, ignore_errors=True)

    if accuracies:
        with (output_dir / "log.txt").open("w") as outfile:
            for accuracy in accuracies:
                outfile.write(f"{accuracy}\n")


def _client_dataset(
    load_dataset,
    path: Path,
    tokenizer,
    template,
    cutoff_len: int,
    train_on_inputs: bool,
    local_val_size: int,
    seed: int,
):
    dataset = load_dataset("json", data_files=str(path))["train"]

    def tokenize(text: str, add_eos_token: bool = True):
        result = tokenizer(text, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if result["input_ids"] and result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def convert(record):
        normalized = normalize_record(record)
        full_prompt = template.format(normalized, include_response=True)
        tokenized = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = template.format(normalized, include_response=False)
            user_len = len(tokenize(user_prompt, add_eos_token=False)["input_ids"])
            tokenized["labels"] = [-100] * user_len + tokenized["labels"][user_len:]
        return tokenized

    if local_val_size > 0:
        split = dataset.train_test_split(test_size=local_val_size, shuffle=True, seed=seed)
        return split["train"].shuffle(seed=seed).map(convert), split["test"].shuffle(seed=seed).map(convert)
    return dataset.shuffle(seed=seed).map(convert), None


def _build_trainer(transformers, model, tokenizer, train_dataset, eval_dataset, output_dir: Path, args: argparse.Namespace):
    gradient_accumulation_steps = max(1, args.local_batch_size // args.micro_batch_size)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=0,
        num_train_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        bf16=args.torch_dtype == "bfloat16",
        fp16=args.torch_dtype == "float16",
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="no",
        eval_steps=200 if eval_dataset is not None else None,
        output_dir=str(output_dir),
        ddp_find_unused_parameters=False if world_size > 1 else None,
        group_by_length=False,
        dataloader_drop_last=False,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    return transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
    )


def _evaluate_mmlu(model, tokenizer, template, eval_path: Path, device: str, GenerationConfig, torch, tqdm) -> float:
    with eval_path.open() as infile:
        records = json.load(infile)
    sampling = GenerationConfig(
        do_sample=True,
        temperature=0.2,
        top_p=0.6,
        top_k=30,
        num_beams=1,
        max_new_tokens=32,
        early_stopping=True,
    )
    right_by_class: dict[str, int] = {}
    total_by_class: dict[str, int] = {}
    response_split = template.response_split

    model.eval()
    for record in tqdm(records, desc="eval"):
        expected = record["output"].replace("The answer is: ", "")
        expected_index, expected_text = expected.split(". ", 1)
        prompt_record = normalize_record(
            {
                "instruction": record["instruction"],
                "input": record.get("input", ""),
                "output": "The answer is: ",
            }
        )
        prompt = template.format(prompt_record, include_response=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                generation_config=sampling,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(generated.sequences[0]).split(response_split)[-1].strip()
        class_name = record["class"]
        total_by_class[class_name] = total_by_class.get(class_name, 0) + 1
        if f"{expected_index}." in answer or expected_text in answer:
            right_by_class[class_name] = right_by_class.get(class_name, 0) + 1

    accuracies = [
        right_by_class.get(class_name, 0) / total
        for class_name, total in total_by_class.items()
        if total
    ]
    return sum(accuracies) / len(accuracies)


def _write_round_metadata(
    round_dir: Path,
    round_id: int,
    variant: str,
    selected: list[int],
    ranks: dict[int, int],
    weights: dict[int, float],
    round_A,
    round_B,
    frozen_A,
    frozen_B,
    args: argparse.Namespace,
) -> None:
    metadata = {
        "round": round_id,
        "variant": variant,
        "selected_clients": selected,
        "client_ranks": {str(key): value for key, value in ranks.items()},
        "client_weights": {str(key): value for key, value in weights.items()},
        "round_rank": next(iter(round_A.values())).shape[0],
        "cumulative_rank": next(iter(frozen_A.values())).shape[0],
        "round_adapter_params": sum(value.numel() for value in round_A.values()) + sum(value.numel() for value in round_B.values()),
        "cumulative_adapter_params": sum(value.numel() for value in frozen_A.values()) + sum(value.numel() for value in frozen_B.values()),
        "rank": args.rank,
        "alpha": args.alpha,
        "heterogeneous": args.heterogeneous,
    }
    with (round_dir / "round_config.json").open("w") as outfile:
        json.dump(metadata, outfile, indent=2)


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_map():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        return {"": int(os.environ.get("LOCAL_RANK") or 0)}
    return "auto"


def _torch_dtype(name: str, torch):
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _ml_imports():
    try:
        import torch
        import transformers
        from datasets import load_dataset
        from tqdm import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch, transformers, datasets, and tqdm. "
            "Install the ML requirements before running this command."
        ) from exc
    return torch, transformers, load_dataset, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, tqdm


if __name__ == "__main__":
    main()
