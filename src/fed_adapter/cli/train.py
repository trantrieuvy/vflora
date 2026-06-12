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
    "nonlinear-ffa": "nonlinear-ffa",
    "nonlinear_ffa": "nonlinear-ffa",
    "ffa": "nonlinear-ffa",
    "nonlinear-rolora": "nonlinear-rolora",
    "nonlinear_rolora": "nonlinear-rolora",
    "rolora": "nonlinear-rolora",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V-FLoRA federated adapter training.")
    parser.add_argument("--task-family", choices=("instruction", "glue"), default="instruction")
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Federated adapter method. Short aliases 'nonlinear', "
            "'cumulative-linear', 'ffa', and 'rolora' are kept for compatibility."
        ),
    )
    parser.add_argument("--method", help="Compatibility alias for --variant, used by FederatedLLM manifests.")
    parser.add_argument("--model", default="tinyllama")
    parser.add_argument("--task-name")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--calibration-path", type=Path)
    parser.add_argument("--num-clients", "--num_clients", type=int, default=10)
    parser.add_argument("--rounds", "--num-communication-rounds", "--num_communication_rounds", type=int, default=3)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--rank", "--lora-r", "--lora_r", type=int, default=16)
    parser.add_argument("--alpha", "--lora-alpha", "--lora_alpha", type=float, default=32)
    parser.add_argument("--dropout", "--lora-dropout", "--lora_dropout", type=float, default=0.05)
    parser.add_argument("--activation", default="gelu", help="Hidden activation for nonlinear FFA/RoLoRA.")
    parser.add_argument("--a-init-std", type=float, default=0.02, help="Seeded A initialization std for nonlinear FFA/RoLoRA.")
    parser.add_argument("--target-modules", "--lora-target-modules", "--lora_target_modules")
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--heter")
    parser.add_argument("--local-ranks", default="64,32,16,16,8,8,4,4,4,4")
    parser.add_argument("--local-batch-size", "--local_batch_size", type=int, default=128)
    parser.add_argument("--micro-batch-size", "--local-micro-batch-size", "--local_micro_batch_size", type=int, default=16)
    parser.add_argument("--local-epochs", "--local-num-epochs", "--local_num_epochs", type=int, default=1)
    parser.add_argument("--learning-rate", "--local-learning-rate", "--local_learning_rate", type=float, default=3e-4)
    parser.add_argument("--local-val-size", "--local-val-set-size", "--local_val_set_size", type=float, default=0)
    parser.add_argument("--local-train-monitor-size", "--local_train_monitor_size", type=int, default=0)
    parser.add_argument("--local-validation-source", "--local_validation_source", default="local_holdout")
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--max-seq-length", "--max_seq_length", dest="cutoff_len", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--prompt-template", default="alpaca")
    parser.add_argument("--train-on-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--warmup-ratio", "--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", "--eval_batch_size", type=int, default=64)
    parser.add_argument("--use-deterministic-algorithms", "--use_deterministic_algorithms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-latest", "--resume_from_latest", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-rounds-per-invocation", "--max_rounds_per_invocation", type=int, default=0)
    parser.add_argument("--retain-adapter-every-n-rounds", "--retain_adapter_every_n_rounds", type=int, default=1)
    parser.add_argument("--keep-local-checkpoints", action="store_true")
    parser.add_argument("--distill-calibration-size", type=int, default=512)
    parser.add_argument("--distill-max-tokens", type=int, default=8192)
    parser.add_argument("--distill-steps", type=int, default=200)
    parser.add_argument("--distill-batch-size", type=int, default=64)
    parser.add_argument("--distill-lr", type=float, default=1e-3)
    parser.add_argument("--distill-weight-decay", type=float, default=0.0)
    parser.add_argument("--distill-max-relative-mse", type=float, default=0.25)
    parser.add_argument("--distill-strict", action="store_true")
    parser.add_argument("--save-distill-teacher", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if _legacy_bool(args.heter):
        args.heterogeneous = True
    if args.task_family == "glue":
        from fed_adapter.cli.train_glue import train as train_glue

        train_glue(args)
        return
    train(args)


def _legacy_bool(value) -> bool:
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


def train(args: argparse.Namespace) -> None:
    torch, transformers, load_dataset, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, tqdm = _ml_imports()
    from fed_adapter.aggregation import (
        aggregate_ffa_b,
        normalized_client_weights,
        stack_linear_lora,
        stack_nonlinear_lora,
        stack_rolora_a_teacher,
        weighted_average,
    )
    from fed_adapter.adapters.ffa import (
        ffa_B_state_dict,
        init_frozen_A,
        init_zero_B,
        inject_ffa_adapters,
        join_ffa_A_state,
        join_ffa_B_state,
        split_ffa_B_state,
    )
    from fed_adapter.adapters.residual import (
        accumulate_adapters,
        adapter_state_dict,
        inject_residual_adapters,
        join_adapter_state,
        split_adapter_state,
    )
    from fed_adapter.adapters.rolora import (
        inject_rolora_adapters,
        join_rolora_state,
        rolora_active_state_dict,
        split_rolora_factor_state,
    )
    from fed_adapter.distillation import distill_nonlinear_lora_modules

    _seed_everything(args.seed, torch)

    data_dir = args.data_root / str(args.num_clients)
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing client data directory: {data_dir}")

    target_modules = [item.strip() for item in (args.target_modules or "q_proj,v_proj").split(",") if item.strip()]
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
    variant_name = args.method or args.variant or "nonlinear-cumulative-flora"
    variant = VARIANT_ALIASES[variant_name]
    _validate_train_args(args, variant)
    ffa = variant == "nonlinear-ffa"
    rolora = variant == "nonlinear-rolora"
    nonlinear = variant == "nonlinear-cumulative-flora"

    global_ffa_rank = max(local_ranks[: args.num_clients]) if args.heterogeneous else args.rank
    ffa_A = None
    ffa_B = None
    if ffa:
        model.to("cpu")
        ffa_A = init_frozen_A(
            model,
            target_modules=target_modules,
            rank=global_ffa_rank,
            seed=args.seed,
            init_std=args.a_init_std,
        )
        ffa_B = init_zero_B(model, target_modules=target_modules, rank=global_ffa_rank)
        torch.save(join_ffa_A_state(ffa_A), output_dir / "A_frozen.bin")
        _write_ffa_config(output_dir, variant, global_ffa_rank, target_modules, args)

    rolora_A = None
    rolora_B = None
    calibration_prompts: list[str] | None = None
    if rolora:
        model.to("cpu")
        rolora_A = init_frozen_A(
            model,
            target_modules=target_modules,
            rank=args.rank,
            seed=args.seed,
            init_std=args.a_init_std,
        )
        rolora_B = init_zero_B(model, target_modules=target_modules, rank=args.rank)
        torch.save(join_rolora_state(rolora_A, rolora_B), output_dir / "adapter_model_initial.bin")
        _write_rolora_config(output_dir, variant, target_modules, args)
        if args.rounds > 1:
            assert args.calibration_path is not None
            calibration_prompts = _load_calibration_prompts(
                args.calibration_path,
                template,
                args.distill_calibration_size,
                args.seed,
            )

    for round_id in tqdm(range(args.rounds), desc="federated rounds"):
        selected = select_clients(args.num_clients, args.client_fraction, seed=round_id)
        client_sizes: dict[int, int] = {}
        client_states = {}
        train_factor = _rolora_train_factor(round_id) if rolora else None

        for client_id in selected:
            rank = local_ranks[client_id] if args.heterogeneous else args.rank
            alpha = frozen_scaling * rank
            model.to("cpu")
            if ffa:
                assert ffa_A is not None and ffa_B is not None
                client_model, adapter_count = inject_ffa_adapters(
                    copy.deepcopy(model),
                    target_modules=target_modules,
                    A_frozen=ffa_A,
                    B_state=ffa_B,
                    scaling=frozen_scaling,
                    dropout=args.dropout,
                    activation=args.activation,
                    client_rank=rank if args.heterogeneous else None,
                )
            elif rolora:
                assert rolora_A is not None and rolora_B is not None and train_factor is not None
                client_model, adapter_count = inject_rolora_adapters(
                    copy.deepcopy(model),
                    target_modules=target_modules,
                    A_state=rolora_A,
                    B_state=rolora_B,
                    scaling=frozen_scaling,
                    dropout=args.dropout,
                    activation=args.activation,
                    train_factor=train_factor,
                )
            else:
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
            if ffa:
                state = ffa_B_state_dict(client_model)
            elif rolora:
                assert train_factor is not None
                state = rolora_active_state_dict(client_model, train_factor)
            else:
                state = adapter_state_dict(client_model)
            checkpoint_dir = output_dir / str(round_id) / f"local_output_{client_id}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, checkpoint_dir / "adapter_model.bin")
            client_sizes[client_id] = len(train_dataset)
            if ffa:
                client_states[client_id] = split_ffa_B_state(state)
            elif rolora:
                assert train_factor is not None
                client_states[client_id] = split_rolora_factor_state(state, train_factor)
            else:
                client_states[client_id] = state
            del trainer, client_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        weights = normalized_client_weights(client_sizes)
        ranks = {client_id: (local_ranks[client_id] if args.heterogeneous else args.rank) for client_id in selected}
        round_dir = output_dir / str(round_id)
        if ffa:
            assert ffa_A is not None and ffa_B is not None
            ffa_B = aggregate_ffa_b(client_states, weights, ffa_B)
            torch.save(join_ffa_B_state(ffa_B), round_dir / "adapter_model.bin")
            _write_ffa_round_metadata(round_dir, round_id, variant, selected, ranks, weights, ffa_A, ffa_B, global_ffa_rank, args)
        elif rolora:
            assert rolora_A is not None and rolora_B is not None and train_factor is not None
            distill_metrics = None
            if train_factor == "B":
                rolora_B = aggregate_ffa_b(client_states, weights, rolora_B)
            else:
                assert calibration_prompts is not None
                teacher_A, teacher_B = stack_rolora_a_teacher(client_states, weights, rolora_B)
                init_A = weighted_average(client_states, weights)
                init_B = {name: value.detach().cpu().clone() for name, value in rolora_B.items()}
                if args.save_distill_teacher:
                    torch.save(join_rolora_state(teacher_A, teacher_B), round_dir / "adapter_model_teacher.bin")
                teacher_model, _ = inject_rolora_adapters(
                    copy.deepcopy(model),
                    target_modules=target_modules,
                    A_state=teacher_A,
                    B_state=teacher_B,
                    scaling=frozen_scaling,
                    dropout=0.0,
                    activation=args.activation,
                    train_factor=None,
                )
                teacher_model.to(args.device)
                activation_inputs = _capture_teacher_inputs(
                    teacher_model,
                    tokenizer,
                    calibration_prompts,
                    target_names=sorted(teacher_A),
                    max_tokens=args.distill_max_tokens,
                    batch_size=args.micro_batch_size,
                    cutoff_len=args.cutoff_len,
                    device=args.device,
                    seed=args.seed + round_id,
                    torch=torch,
                )
                rolora_A, rolora_B, distill_metrics = distill_nonlinear_lora_modules(
                    activation_inputs,
                    teacher_A,
                    teacher_B,
                    init_A,
                    init_B,
                    activation=args.activation,
                    steps=args.distill_steps,
                    batch_size=args.distill_batch_size,
                    lr=args.distill_lr,
                    weight_decay=args.distill_weight_decay,
                    max_relative_mse=args.distill_max_relative_mse,
                    strict=args.distill_strict,
                    seed=args.seed + round_id,
                    device=args.device,
                )
                del teacher_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            torch.save(join_rolora_state(rolora_A, rolora_B), round_dir / "adapter_model.bin")
            _write_rolora_round_metadata(
                round_dir,
                round_id,
                variant,
                selected,
                ranks,
                weights,
                train_factor,
                rolora_A,
                rolora_B,
                distill_metrics,
                args,
                calibration_count=len(calibration_prompts or []),
            )
        elif nonlinear:
            round_state = stack_nonlinear_lora(client_states, weights, ranks)
            round_A, round_B = split_adapter_state(round_state)
            frozen_A, frozen_B = accumulate_adapters(frozen_A, frozen_B, round_A, round_B)
            cumulative_state = join_adapter_state(frozen_A, frozen_B)
            torch.save(round_state, round_dir / "adapter_model_delta.bin")
            torch.save(cumulative_state, round_dir / "adapter_model.bin")
            _write_round_metadata(round_dir, round_id, variant, selected, ranks, weights, round_A, round_B, frozen_A, frozen_B, args)
        else:
            round_state = stack_linear_lora(client_states, weights, ranks)
            round_A, round_B = split_adapter_state(round_state)
            frozen_A, frozen_B = accumulate_adapters(frozen_A, frozen_B, round_A, round_B)
            cumulative_state = join_adapter_state(frozen_A, frozen_B)
            torch.save(round_state, round_dir / "adapter_model_delta.bin")
            torch.save(cumulative_state, round_dir / "adapter_model.bin")
            _write_round_metadata(round_dir, round_id, variant, selected, ranks, weights, round_A, round_B, frozen_A, frozen_B, args)

        if args.eval_path:
            if ffa:
                assert ffa_A is not None and ffa_B is not None
                eval_model, _ = inject_ffa_adapters(
                    copy.deepcopy(model),
                    target_modules=target_modules,
                    A_frozen=ffa_A,
                    B_state=ffa_B,
                    scaling=frozen_scaling,
                    dropout=0.0,
                    activation=args.activation,
                    client_rank=None,
                )
            elif rolora:
                assert rolora_A is not None and rolora_B is not None
                eval_model, _ = inject_rolora_adapters(
                    copy.deepcopy(model),
                    target_modules=target_modules,
                    A_state=rolora_A,
                    B_state=rolora_B,
                    scaling=frozen_scaling,
                    dropout=0.0,
                    activation=args.activation,
                    train_factor=None,
                )
            else:
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
            print(f"Acc round {round_id}: {accuracy}")
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


def _validate_train_args(args: argparse.Namespace, variant: str) -> None:
    if variant != "nonlinear-rolora":
        return
    if args.heterogeneous:
        raise ValueError("nonlinear-rolora currently supports homogeneous rank only")
    if args.calibration_path is not None and args.calibration_path.name == "global_test.json":
        raise ValueError("nonlinear-rolora distillation must not use global_test.json")
    if args.rounds > 1 and args.calibration_path is None:
        raise ValueError("--calibration-path is required for nonlinear-rolora A-round distillation")
    if args.distill_calibration_size <= 0:
        raise ValueError("--distill-calibration-size must be positive")
    if args.distill_max_tokens <= 0:
        raise ValueError("--distill-max-tokens must be positive")
    if args.distill_steps < 0:
        raise ValueError("--distill-steps cannot be negative")
    if args.distill_batch_size <= 0:
        raise ValueError("--distill-batch-size must be positive")


def _rolora_train_factor(round_id: int) -> str:
    return "B" if round_id % 2 == 0 else "A"


def _load_calibration_prompts(
    path: Path,
    template,
    limit: int,
    seed: int,
) -> list[str]:
    records = _read_records(path)
    rng = random.Random(seed)
    rng.shuffle(records)
    prompts: list[str] = []
    for record in records:
        try:
            prompt = _calibration_prompt(record, template)
        except ValueError:
            continue
        if prompt.strip():
            prompts.append(prompt)
        if len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(f"No usable calibration prompts found in {path}")
    return prompts


def _read_records(path: Path) -> list[dict[str, object]]:
    text = path.read_text().strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            for key in ("records", "data", "examples"):
                if isinstance(parsed.get(key), list):
                    records = parsed[key]
                    break
            else:
                records = [parsed]
        else:
            raise ValueError(f"Unsupported calibration JSON root in {path}")
    return [record for record in records if isinstance(record, dict)]


def _calibration_prompt(record: dict[str, object], template) -> str:
    if "instruction" not in record:
        raise ValueError("calibration record is missing instruction")
    prompt_record = {
        "instruction": record["instruction"],
        "input": record.get("input", record.get("context", "")),
        "output": "",
    }
    normalized = normalize_record(prompt_record)
    return template.format(normalized, include_response=False)


def _capture_teacher_inputs(
    teacher_model,
    tokenizer,
    prompts: list[str],
    *,
    target_names: list[str],
    max_tokens: int,
    batch_size: int,
    cutoff_len: int,
    device: str,
    seed: int,
    torch,
) -> dict[str, object]:
    captured: dict[str, list[object]] = {name: [] for name in target_names}
    counts = {name: 0 for name in target_names}
    generators = {}
    for index, name in enumerate(target_names):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + index)
        generators[name] = generator

    current_mask = None
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0].detach().to(device="cpu", dtype=torch.float32)
            if x.ndim == 3 and current_mask is not None and tuple(current_mask.shape) == tuple(x.shape[:2]):
                rows = x[current_mask.to(device="cpu", dtype=torch.bool)]
            else:
                rows = x.reshape(-1, x.shape[-1])
            if rows.numel() == 0:
                return
            _append_limited_rows(captured, counts, name, rows, max_tokens, generators[name], torch)

        return hook

    target_set = set(target_names)
    for name, module in teacher_model.named_modules():
        if name in target_set:
            handles.append(module.register_forward_pre_hook(make_hook(name)))
    if len(handles) != len(target_names):
        found = {name for name, module in teacher_model.named_modules() if name in target_set}
        missing = sorted(target_set - found)
        raise ValueError(f"Could not register calibration hooks for target modules: {missing}")

    teacher_model.eval()
    try:
        for start in range(0, len(prompts), max(1, batch_size)):
            batch_prompts = prompts[start : start + max(1, batch_size)]
            encoded = tokenizer(
                batch_prompts,
                truncation=True,
                max_length=cutoff_len,
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in encoded.items()}
            current_mask = batch.get("attention_mask")
            with torch.no_grad():
                teacher_model(**batch)
    finally:
        for handle in handles:
            handle.remove()

    result = {
        name: _finalize_limited_rows(captured[name], max_tokens, generators[name], torch)
        for name in target_names
    }
    empty = [name for name, value in result.items() if value.shape[0] == 0]
    if empty:
        raise ValueError(f"No calibration activations captured for target modules: {empty}")
    return result


def _append_limited_rows(captured, counts, name, rows, max_tokens, generator, torch) -> None:
    captured[name].append(rows.detach().cpu())
    counts[name] += rows.shape[0]
    if counts[name] <= max_tokens * 2:
        return
    merged = torch.cat(captured[name], dim=0)
    if merged.shape[0] > max_tokens:
        indices = torch.randperm(merged.shape[0], generator=generator)[:max_tokens]
        merged = merged.index_select(0, indices)
    captured[name] = [merged.contiguous()]
    counts[name] = merged.shape[0]


def _finalize_limited_rows(parts, max_tokens, generator, torch):
    if not parts:
        return torch.empty(0, 0)
    merged = torch.cat(parts, dim=0)
    if merged.shape[0] > max_tokens:
        indices = torch.randperm(merged.shape[0], generator=generator)[:max_tokens]
        merged = merged.index_select(0, indices)
    return merged.contiguous()


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


def _write_ffa_config(
    output_dir: Path,
    variant: str,
    global_rank: int,
    target_modules: list[str],
    args: argparse.Namespace,
) -> None:
    metadata = {
        "variant": variant,
        "model": args.model,
        "num_clients": args.num_clients,
        "rank": args.rank,
        "global_rank": global_rank,
        "alpha": args.alpha,
        "effective_scaling": args.alpha / args.rank,
        "dropout": args.dropout,
        "target_modules": target_modules,
        "activation": args.activation,
        "a_init_std": args.a_init_std,
        "a_init_seed": args.seed,
        "heterogeneous": args.heterogeneous,
        "local_ranks": [int(rank) for rank in args.local_ranks.split(",") if rank.strip()]
        if args.heterogeneous
        else None,
    }
    with (output_dir / "ffa_config.json").open("w") as outfile:
        json.dump(metadata, outfile, indent=2)


def _write_rolora_config(
    output_dir: Path,
    variant: str,
    target_modules: list[str],
    args: argparse.Namespace,
) -> None:
    metadata = {
        "variant": variant,
        "model": args.model,
        "num_clients": args.num_clients,
        "rank": args.rank,
        "alpha": args.alpha,
        "effective_scaling": args.alpha / args.rank,
        "dropout": args.dropout,
        "target_modules": target_modules,
        "activation": args.activation,
        "a_init_std": args.a_init_std,
        "a_init_seed": args.seed,
        "heterogeneous": False,
        "round_schedule": "B,A alternating from round 0",
        "calibration_path": str(args.calibration_path) if args.calibration_path else None,
        "distill_calibration_size": args.distill_calibration_size,
        "distill_max_tokens": args.distill_max_tokens,
        "distill_steps": args.distill_steps,
        "distill_batch_size": args.distill_batch_size,
        "distill_lr": args.distill_lr,
        "distill_weight_decay": args.distill_weight_decay,
        "distill_max_relative_mse": args.distill_max_relative_mse,
        "distill_strict": args.distill_strict,
    }
    with (output_dir / "rolora_config.json").open("w") as outfile:
        json.dump(metadata, outfile, indent=2)


def _write_ffa_round_metadata(
    round_dir: Path,
    round_id: int,
    variant: str,
    selected: list[int],
    ranks: dict[int, int],
    weights: dict[int, float],
    frozen_A,
    global_B,
    global_rank: int,
    args: argparse.Namespace,
) -> None:
    metadata = {
        "round": round_id,
        "variant": variant,
        "selected_clients": selected,
        "client_ranks": {str(key): value for key, value in ranks.items()},
        "client_weights": {str(key): value for key, value in weights.items()},
        "round_rank": global_rank,
        "global_rank": global_rank,
        "rank_semantics": "ffa_global_B",
        "frozen_A_params": sum(value.numel() for value in frozen_A.values()),
        "trainable_B_params": sum(value.numel() for value in global_B.values()),
        "rank": args.rank,
        "alpha": args.alpha,
        "effective_scaling": args.alpha / args.rank,
        "activation": args.activation,
        "a_init_std": args.a_init_std,
        "heterogeneous": args.heterogeneous,
    }
    with (round_dir / "round_config.json").open("w") as outfile:
        json.dump(metadata, outfile, indent=2)


def _write_rolora_round_metadata(
    round_dir: Path,
    round_id: int,
    variant: str,
    selected: list[int],
    ranks: dict[int, int],
    weights: dict[int, float],
    train_factor: str,
    global_A,
    global_B,
    distill_metrics,
    args: argparse.Namespace,
    calibration_count: int,
) -> None:
    distillation_ran = distill_metrics is not None
    metadata = {
        "round": round_id,
        "variant": variant,
        "selected_clients": selected,
        "client_ranks": {str(key): value for key, value in ranks.items()},
        "client_weights": {str(key): value for key, value in weights.items()},
        "trained_factor": train_factor,
        "round_rank": args.rank,
        "global_rank": args.rank,
        "rank_semantics": "fixed_rolora_rank",
        "global_A_params": sum(value.numel() for value in global_A.values()),
        "global_B_params": sum(value.numel() for value in global_B.values()),
        "rank": args.rank,
        "alpha": args.alpha,
        "effective_scaling": args.alpha / args.rank,
        "activation": args.activation,
        "a_init_std": args.a_init_std,
        "heterogeneous": False,
        "distillation_ran": distillation_ran,
        "calibration_path": str(args.calibration_path) if args.calibration_path else None,
        "calibration_records": calibration_count if distillation_ran else 0,
        "distill_metrics": distill_metrics,
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
