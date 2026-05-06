import argparse
import random
from collections import OrderedDict
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch

import attack_weight_output_utils as weight_utils
import federated_cifar10_hist_mingaplr_base as hist_base

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SIBLING_CIFAR10_DIR = REPO_ROOT / "1.accuracy_analysis" / "cifar10"
SIBLING_ADAPT_CIFAR_DIR = REPO_ROOT / "4.adaptability_analysis" / "cifar-10"
for extra_path in (REPO_ROOT, SIBLING_CIFAR10_DIR, SIBLING_ADAPT_CIFAR_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

import federated_cifar10_mingaplr_compare as base
import federated_cifar10_mingaplr_dirichlet05_quality_perf_compare as perf_base
from federated_cifar10_hist_mingaplr_dirichlet_fixed_noise_common import (
    FixedLabelSubset,
    build_fixed_noisy_labels,
    equal_size_dirichlet_partition,
)


DIRICHLET_ALPHA = 0.5
DEFAULT_NOISE20_CLIENTS = 5
DEFAULT_NOISE40_CLIENTS = 5
DEFAULT_NOISE60_CLIENTS = 5
DEFAULT_MALICIOUS_CLIENTS = 10
RATE_LABELS = {
    0.0: "clean",
    0.2: "noise20",
    0.4: "noise40",
    0.6: "noise60",
}


@dataclass(frozen=True)
class AttackSpec:
    attack_type: str
    display_name: str
    code_version: str
    default_output_json: Path
    description: str
    add_attack_args: Callable[[argparse.ArgumentParser], None]
    validate_attack_args: Callable[[argparse.Namespace], None]
    attack_config: Callable[[argparse.Namespace], Dict[str, float]]
    apply_attack: Callable[[dict, dict, argparse.Namespace, dict, int, int], OrderedDict]


def parse_args(attack_spec):
    parser = argparse.ArgumentParser(description=attack_spec.description)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-json", type=Path, default=attack_spec.default_output_json)
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--proxy-batch-size", type=int, default=1000)
    parser.add_argument("--task-size", type=int, default=1000)
    parser.add_argument("--gen-test-size", type=int, default=10000)
    parser.add_argument("--proxy-batches-per-round", type=int, default=1)
    parser.add_argument("--local-optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--local-lr", type=float, default=0.01)
    parser.add_argument("--local-momentum", type=float, default=0.9)
    parser.add_argument("--local-weight-decay", type=float, default=0.0)
    parser.add_argument("--server-lr-cap", type=float, default=5.0)
    parser.add_argument("--server-opt-steps", type=int, default=10)
    parser.add_argument("--server-opt-weight-tol", type=float, default=1e-4)
    parser.add_argument("--grad-damping", type=float, default=base.DEFAULT_GRAD_DAMPING)
    parser.add_argument(
        "--num-malicious-clients",
        type=int,
        default=DEFAULT_MALICIOUS_CLIENTS,
        help="The last clients are treated as Byzantine attackers",
    )
    parser.add_argument(
        "--noise20-clients",
        type=int,
        default=DEFAULT_NOISE20_CLIENTS,
        help="Number of benign clients with 20 percent fixed label noise",
    )
    parser.add_argument(
        "--noise40-clients",
        type=int,
        default=DEFAULT_NOISE40_CLIENTS,
        help="Number of benign clients with 40 percent fixed label noise",
    )
    parser.add_argument(
        "--noise60-clients",
        type=int,
        default=DEFAULT_NOISE60_CLIENTS,
        help="Number of benign clients with 60 percent fixed label noise",
    )
    attack_spec.add_attack_args(parser)
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def validate_common_args(args, attack_spec):
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be <= num_clients")
    if args.proxy_batches_per_round <= 0:
        raise ValueError("proxy_batches_per_round must be positive")
    if args.task_size <= 0 or args.gen_test_size <= 0:
        raise ValueError("task_size and gen_test_size must be positive")
    if args.num_malicious_clients < 0 or args.num_malicious_clients > args.num_clients:
        raise ValueError("num_malicious_clients must be in [0, num_clients]")
    for option_name in ("noise20_clients", "noise40_clients", "noise60_clients"):
        if getattr(args, option_name) < 0:
            raise ValueError(f"{option_name} must be non-negative")
    noisy_client_count = args.noise20_clients + args.noise40_clients + args.noise60_clients
    if noisy_client_count + args.num_malicious_clients > args.num_clients:
        raise ValueError("noise clients plus malicious clients must not exceed num_clients")
    attack_spec.validate_attack_args(args)


def assert_state_is_finite(state_dict, context):
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite tensor detected in {context}: key={key}")


def build_client_metadata(args):
    clean_count = (
        args.num_clients
        - args.noise20_clients
        - args.noise40_clients
        - args.noise60_clients
        - args.num_malicious_clients
    )
    noise20_start = clean_count
    noise40_start = noise20_start + args.noise20_clients
    noise60_start = noise40_start + args.noise40_clients
    attack_start = noise60_start + args.noise60_clients

    client_metadata = []
    malicious_client_ids = []
    for client_id in range(args.num_clients):
        if client_id >= attack_start:
            category = "attack"
            noise_rate = 0.0
            is_malicious = True
            malicious_client_ids.append(client_id)
        elif client_id >= noise60_start:
            noise_rate = 0.6
            category = RATE_LABELS[noise_rate]
            is_malicious = False
        elif client_id >= noise40_start:
            noise_rate = 0.4
            category = RATE_LABELS[noise_rate]
            is_malicious = False
        elif client_id >= noise20_start:
            noise_rate = 0.2
            category = RATE_LABELS[noise_rate]
            is_malicious = False
        else:
            noise_rate = 0.0
            category = RATE_LABELS[noise_rate]
            is_malicious = False

        client_metadata.append(
            {
                "client_category": category,
                "noise_rate": float(noise_rate),
                "is_malicious": bool(is_malicious),
            }
        )

    return client_metadata, malicious_client_ids


def build_mixed_noise_client_loaders(dataset, full_targets, client_indices, client_metadata, batch_size, loader_kwargs):
    client_loaders = []
    client_sizes = []
    full_targets = np.asarray(full_targets, dtype=np.int64)

    for client_id, indices in enumerate(client_indices):
        noise_rate = float(client_metadata[client_id].get("noise_rate", 0.0))
        noisy_labels = build_fixed_noisy_labels(
            full_targets[np.asarray(indices, dtype=np.int64)],
            noise_rate,
            base.FIXED_SEED + client_id,
        )
        subset = FixedLabelSubset(dataset, indices, noisy_labels)
        generator = torch.Generator()
        generator.manual_seed(base.FIXED_SEED + client_id)
        client_loader_kwargs = dict(loader_kwargs)
        client_loader_kwargs["generator"] = generator
        client_loaders.append(base.DataLoader(subset, batch_size=batch_size, **client_loader_kwargs))
        client_sizes.append(len(indices))

    return client_loaders, client_sizes


def build_shared_experiment_state(args):
    base.set_seed(base.FIXED_SEED)
    devices = base.resolve_devices(args)
    device = devices[0]

    train_dataset, test_dataset = base.load_cifar10(args.data_dir)
    full_dataset = base.ConcatDataset([train_dataset, test_dataset])
    full_targets = np.asarray(train_dataset.targets + test_dataset.targets, dtype=np.int64)
    private_indices, task_indices, gen_test_indices = base.stratified_three_way_split_indices(
        full_targets,
        args.task_size,
        args.gen_test_size,
        base.FIXED_SEED,
    )

    private_targets = full_targets[private_indices]
    client_indices, unused_private_indices, per_client_private_size = equal_size_dirichlet_partition(
        private_indices,
        private_targets,
        args.num_clients,
        DIRICHLET_ALPHA,
        base.FIXED_SEED,
    )

    schedule_rng = random.Random(base.FIXED_SEED)
    selected_clients_schedule = [
        schedule_rng.sample(range(args.num_clients), args.clients_per_round)
        for _ in range(args.rounds)
    ]

    client_metadata, malicious_client_ids = build_client_metadata(args)

    base.set_seed(base.FIXED_SEED)
    initial_model = base.build_model().to(device)
    initial_global_state = base.clone_state_dict_to_cpu(initial_model.state_dict())
    parameter_names = set(base.get_param_names(initial_model))

    return {
        "device": device,
        "full_dataset": full_dataset,
        "full_targets": full_targets,
        "private_indices": private_indices,
        "task_indices": task_indices,
        "gen_test_indices": gen_test_indices,
        "client_indices": client_indices,
        "selected_clients_schedule": selected_clients_schedule,
        "unused_private_indices": unused_private_indices,
        "per_client_private_size": per_client_private_size,
        "malicious_client_ids": malicious_client_ids,
        "client_metadata": client_metadata,
        "initial_global_state": initial_global_state,
        "parameter_names": parameter_names,
    }


def build_run_loaders(args, shared_state):
    train_loader_kwargs = base.build_loader_kwargs(args, shuffle=True)
    eval_loader_kwargs = base.build_loader_kwargs(args, shuffle=False)

    client_loader_kwargs = dict(train_loader_kwargs)
    client_loader_kwargs["num_workers"] = 0
    client_loader_kwargs.pop("persistent_workers", None)

    client_loaders, client_sizes = build_mixed_noise_client_loaders(
        shared_state["full_dataset"],
        shared_state["full_targets"],
        shared_state["client_indices"],
        shared_state["client_metadata"],
        args.batch_size,
        client_loader_kwargs,
    )

    task_subset = base.Subset(shared_state["full_dataset"], shared_state["task_indices"].tolist())
    proxy_batch_size = min(args.proxy_batch_size, len(task_subset))
    task_loader = base.DataLoader(task_subset, batch_size=proxy_batch_size, **train_loader_kwargs)
    task_iterator = iter(task_loader)
    task_eval_loader = base.DataLoader(task_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    gen_test_subset = base.Subset(shared_state["full_dataset"], shared_state["gen_test_indices"].tolist())
    gen_test_loader = base.DataLoader(gen_test_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    return {
        "client_loaders": client_loaders,
        "client_sizes": client_sizes,
        "task_loader": task_loader,
        "task_iterator": task_iterator,
        "task_eval_loader": task_eval_loader,
        "gen_test_loader": gen_test_loader,
        "proxy_batch_size": proxy_batch_size,
    }


def get_selected_client_type_summary(selected_clients, client_metadata):
    selected_types = []
    for client_id in selected_clients:
        category = client_metadata[client_id].get("client_category", "unknown")
        if category != "clean":
            selected_types.append(f"{client_id}:{category}")
    return selected_types


def maybe_attack_local_state(client_id, round_idx, global_state, clean_local_state, args, attack_spec, shared_state):
    if client_id not in set(shared_state["malicious_client_ids"]):
        return clean_local_state, False
    attacked_state = attack_spec.apply_attack(
        global_state,
        clean_local_state,
        args,
        shared_state,
        client_id,
        round_idx,
    )
    assert_state_is_finite(attacked_state, "attacked_local_state")
    return attacked_state, True


def train_selected_clients_with_attack(round_idx, selected_clients, global_model, loaders, args, attack_spec, shared_state):
    device = shared_state["device"]
    global_snapshot = base.clone_state_dict_to_cpu(global_model.state_dict())
    malicious_client_id_set = set(shared_state["malicious_client_ids"])

    local_states = []
    sample_counts = []
    selected_malicious_clients = []
    for client_id in selected_clients:
        local_seed = base.FIXED_SEED + round_idx * 1000 + client_id
        clean_local_state, sample_count = base.client_local_train(
            global_snapshot,
            loaders["client_loaders"][client_id],
            loaders["client_sizes"][client_id],
            device,
            args,
            local_seed,
        )
        attacked_local_state, was_attacked = maybe_attack_local_state(
            client_id,
            round_idx,
            global_snapshot,
            clean_local_state,
            args,
            attack_spec,
            shared_state,
        )
        if was_attacked:
            selected_malicious_clients.append(client_id)
        elif client_id in malicious_client_id_set:
            raise RuntimeError(f"Client {client_id} was marked malicious but attack was not applied")
        local_states.append(attacked_local_state)
        sample_counts.append(sample_count)

    return local_states, sample_counts, selected_malicious_clients


def build_ours_round_record(
    round_idx,
    metrics,
    selected_clients,
    selected_malicious_clients,
    smartfl_result,
    client_records,
    attack_spec,
):
    return {
        "round_index": int(round_idx),
        "attack_name": attack_spec.attack_type,
        "selected_clients": list(selected_clients),
        "selected_malicious_clients": list(selected_malicious_clients),
        "selected_malicious_count": len(selected_malicious_clients),
        "proxy_loss": float(smartfl_result["proxy_loss"]),
        "task_loss": float(metrics["task_loss"]),
        "task_acc": float(metrics["task_acc"]),
        "gen_loss": float(metrics["gen_loss"]),
        "gen_acc": float(metrics["gen_acc"]),
        "server_steps_used": int(smartfl_result["steps_used"]),
        "server_early_stopped": bool(smartfl_result["early_stopped"]),
        "server_stop_reason": smartfl_result["stop_reason"],
        "server_final_change_l1": float(smartfl_result["final_step_change_l1"]),
        "server_final_change_max_abs": float(smartfl_result["final_step_change_max_abs"]),
        "clients": client_records,
    }


def build_fedavg_accuracy_record(round_idx, metrics, selected_malicious_clients, attack_spec):
    return {
        "round_index": int(round_idx),
        "attack_name": attack_spec.attack_type,
        "selected_malicious_count": len(selected_malicious_clients),
        "task_acc": float(metrics["task_acc"]),
        "gen_acc": float(metrics["gen_acc"]),
    }


def run_ours_under_noise_attack(args, shared_state, attack_spec):
    device = shared_state["device"]
    loaders = build_run_loaders(args, shared_state)

    global_model = base.build_model().to(device)
    global_model.load_state_dict(shared_state["initial_global_state"])
    task_iterator = loaders["task_iterator"]
    round_records = []

    for round_idx, selected_clients in enumerate(shared_state["selected_clients_schedule"], start=1):
        selected_types = get_selected_client_type_summary(selected_clients, shared_state["client_metadata"])
        print(
            f"[FedAvg + Ours + Noise + {attack_spec.display_name}] round {round_idx:03d} "
            f"selected_clients={selected_clients} non_clean_selected={selected_types}"
        )

        proxy_batches, task_iterator = base.next_batches(
            loaders["task_loader"],
            task_iterator,
            args.proxy_batches_per_round,
        )
        local_states, sample_counts, selected_malicious_clients = train_selected_clients_with_attack(
            round_idx,
            selected_clients,
            global_model,
            loaders,
            args,
            attack_spec,
            shared_state,
        )

        smartfl_result = weight_utils.optimize_smartfl_weights_no_early_stop_with_analysis(
            global_model,
            local_states,
            sample_counts,
            selected_clients,
            proxy_batches,
            args.server_lr_cap,
            args.server_opt_steps,
            args.server_opt_weight_tol,
            args.grad_damping,
            device,
            compute_proxy_loss_and_grad_for_alpha=base.compute_proxy_loss_and_grad_for_alpha,
            evaluate_proxy_loss_for_alpha=base.evaluate_proxy_loss_for_alpha,
            normalize_raw_weights=base.normalize_raw_weights,
        )
        loo_full_proxy_loss, loo_losses = hist_base.compute_leave_one_out_proxy_metrics(
            global_model,
            local_states,
            smartfl_result["base_alpha"],
            proxy_batches,
            device,
        )
        next_state = base.aggregate_states_on_cpu(local_states, smartfl_result["alpha"], global_model)
        assert_state_is_finite(next_state, f"ours_aggregated_global_state_round_{round_idx}")
        global_model.load_state_dict(next_state)

        metrics = perf_base.evaluate_global_model(
            global_model,
            loaders["task_eval_loader"],
            loaders["gen_test_loader"],
            device,
        )
        client_records = hist_base.build_client_analysis_records(
            selected_clients,
            sample_counts,
            smartfl_result["alpha"],
            smartfl_result["mean_gradients"],
            loo_full_proxy_loss,
            loo_losses,
            shared_state["client_metadata"],
        )
        round_record = build_ours_round_record(
            round_idx,
            metrics,
            selected_clients,
            selected_malicious_clients,
            smartfl_result,
            client_records,
            attack_spec,
        )
        round_records.append(round_record)
        print(
            f"[FedAvg + Ours + Noise + {attack_spec.display_name}] round {round_idx:03d} done: "
            f"attack_count={len(selected_malicious_clients)} "
            f"task_acc={metrics['task_acc']:.4f} gen_acc={metrics['gen_acc']:.4f}"
        )

    return round_records


def run_fedavg_under_noise_attack(args, shared_state, attack_spec):
    device = shared_state["device"]
    loaders = build_run_loaders(args, shared_state)

    global_model = base.build_model().to(device)
    global_model.load_state_dict(shared_state["initial_global_state"])
    round_records = []

    for round_idx, selected_clients in enumerate(shared_state["selected_clients_schedule"], start=1):
        selected_types = get_selected_client_type_summary(selected_clients, shared_state["client_metadata"])
        print(
            f"[FedAvg + Noise + {attack_spec.display_name}] round {round_idx:03d} "
            f"selected_clients={selected_clients} non_clean_selected={selected_types}"
        )
        local_states, sample_counts, selected_malicious_clients = train_selected_clients_with_attack(
            round_idx,
            selected_clients,
            global_model,
            loaders,
            args,
            attack_spec,
            shared_state,
        )
        total_samples = float(sum(sample_counts))
        aggregation_weights = [sample_count / total_samples for sample_count in sample_counts]
        next_state = base.aggregate_states_on_cpu(local_states, aggregation_weights, global_model)
        assert_state_is_finite(next_state, f"fedavg_aggregated_global_state_round_{round_idx}")
        global_model.load_state_dict(next_state)

        metrics = perf_base.evaluate_global_model(
            global_model,
            loaders["task_eval_loader"],
            loaders["gen_test_loader"],
            device,
        )
        round_records.append(
            build_fedavg_accuracy_record(round_idx, metrics, selected_malicious_clients, attack_spec)
        )
        print(
            f"[FedAvg + Noise + {attack_spec.display_name}] round {round_idx:03d} done: "
            f"attack_count={len(selected_malicious_clients)} "
            f"task_acc={metrics['task_acc']:.4f} gen_acc={metrics['gen_acc']:.4f}"
        )

    return round_records


def build_comparison_summary(fedavg_rounds, ours_rounds):
    if not fedavg_rounds or not ours_rounds:
        return {}

    fedavg_final = fedavg_rounds[-1]
    ours_final = ours_rounds[-1]
    return {
        "final_task_acc_fedavg": float(fedavg_final["task_acc"]),
        "final_task_acc_ours": float(ours_final["task_acc"]),
        "final_gen_acc_fedavg": float(fedavg_final["gen_acc"]),
        "final_gen_acc_ours": float(ours_final["gen_acc"]),
        "final_task_acc_gain": float(ours_final["task_acc"] - fedavg_final["task_acc"]),
        "final_gen_acc_gain": float(ours_final["gen_acc"] - fedavg_final["gen_acc"]),
        "best_task_acc_fedavg": float(max(item["task_acc"] for item in fedavg_rounds)),
        "best_task_acc_ours": float(max(item["task_acc"] for item in ours_rounds)),
        "best_gen_acc_fedavg": float(max(item["gen_acc"] for item in fedavg_rounds)),
        "best_gen_acc_ours": float(max(item["gen_acc"] for item in ours_rounds)),
    }


def build_client_group_config(args, malicious_client_ids):
    clean_count = (
        args.num_clients
        - args.noise20_clients
        - args.noise40_clients
        - args.noise60_clients
        - args.num_malicious_clients
    )
    return {
        "clean_clients": clean_count,
        "noise20_clients": args.noise20_clients,
        "noise40_clients": args.noise40_clients,
        "noise60_clients": args.noise60_clients,
        "num_malicious_clients": args.num_malicious_clients,
        "malicious_client_ids": malicious_client_ids,
        "client_type_layout": (
            "clean first, then noise20, noise40, noise60, and attack clients last; "
            "attack clients use clean local labels before Byzantine update manipulation"
        ),
    }


def run_experiment(args, attack_spec):
    validate_common_args(args, attack_spec)
    shared_state = build_shared_experiment_state(args)

    print(
        "CIFAR-10 Dirichlet-0.5 noise+attack contribution evaluation setup:",
        f"code_version={attack_spec.code_version}",
        f"base_code_version={base.CODE_VERSION}",
        "dataset=CIFAR-10",
        "partition=Dirichlet",
        f"dirichlet_alpha={DIRICHLET_ALPHA}",
        "client_data_quality=mixed_clean_fixed_label_noise_and_attack",
        f"attack_type={attack_spec.attack_type}",
        *[f"{key}={value}" for key, value in attack_spec.attack_config(args).items()],
        "same_partition_same_schedule_same_init=true",
        "run_order=ours_then_fedavg",
        "fedavg_branch_records=accuracy_only",
        "server_weight_optimization=no_early_stop_for_attack_experiments",
        f"server_lr_cap={args.server_lr_cap}",
        f"grad_damping={args.grad_damping}",
        f"clients={args.num_clients}",
        f"clients_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"malicious_client_ids={shared_state['malicious_client_ids']}",
        f"private_train_size={len(shared_state['private_indices'])}",
        f"per_client_private_size={shared_state['per_client_private_size']}",
        f"unused_private_samples={len(shared_state['unused_private_indices'])}",
        f"task_size={len(shared_state['task_indices'])}",
        f"gen_test_size={len(shared_state['gen_test_indices'])}",
        f"proxy_batch_size={args.proxy_batch_size}",
        f"proxy_batches_per_round={args.proxy_batches_per_round}",
        f"fixed_seed={base.FIXED_SEED}",
        f"device={shared_state['device']}",
    )

    ours_rounds = run_ours_under_noise_attack(args, shared_state, attack_spec)
    fedavg_rounds = run_fedavg_under_noise_attack(args, shared_state, attack_spec)

    payload = {
        "config": {
            "code_version": attack_spec.code_version,
            "base_code_version": base.CODE_VERSION,
            "dataset": "CIFAR-10",
            "partition": "Dirichlet",
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "client_data_quality": "mixed_clean_fixed_label_noise_and_attack",
            "attack_type": attack_spec.attack_type,
            **attack_spec.attack_config(args),
            "same_partition_same_schedule_same_init": True,
            "run_order": "ours_then_fedavg",
            "fedavg_branch_records": "accuracy_only",
            "num_clients": args.num_clients,
            "clients_per_round": args.clients_per_round,
            "rounds": args.rounds,
            **build_client_group_config(args, shared_state["malicious_client_ids"]),
            "task_size": len(shared_state["task_indices"]),
            "gen_test_size": len(shared_state["gen_test_indices"]),
            "private_train_size": len(shared_state["private_indices"]),
            "per_client_private_size": shared_state["per_client_private_size"],
            "unused_private_samples": len(shared_state["unused_private_indices"]),
            "proxy_batch_size": args.proxy_batch_size,
            "proxy_batches_per_round": args.proxy_batches_per_round,
            "grad_damping": args.grad_damping,
            "server_lr_cap": args.server_lr_cap,
            "server_opt_steps": args.server_opt_steps,
            "fixed_seed": base.FIXED_SEED,
            "window_size": hist_base.WINDOW_SIZE,
            "methods": {
                "ours": f"fedavg_plus_server_side_quality_aware_weight_optimization_under_{attack_spec.attack_type}_and_fixed_noise",
                "fedavg": f"fedavg_under_{attack_spec.attack_type}_and_fixed_noise_accuracy_only",
            },
        },
        "rounds": ours_rounds,
        "rounds_detail": ours_rounds,
        "ours_rounds": ours_rounds,
        "fedavg_rounds": fedavg_rounds,
        "fedavg_accuracy_rounds": fedavg_rounds,
        "window_average_weights": hist_base.build_window_average_weights(
            ours_rounds,
            hist_base.WINDOW_SIZE,
        ),
        "ours_window_average_weights": hist_base.build_window_average_weights(
            ours_rounds,
            hist_base.WINDOW_SIZE,
        ),
        "client_average_weights": hist_base.build_client_average_weights(
            ours_rounds,
            args.num_clients,
            args.rounds,
            shared_state["client_metadata"],
        ),
        "ours_client_average_weights": hist_base.build_client_average_weights(
            ours_rounds,
            args.num_clients,
            args.rounds,
            shared_state["client_metadata"],
        ),
        "comparison_summary": build_comparison_summary(fedavg_rounds, ours_rounds),
    }

    output_path = base.resolve_output_path(args.output_json)
    base.write_json_snapshot(payload, output_path)
    print(f"{attack_spec.display_name} noise+attack contribution JSON saved to: {output_path}")
