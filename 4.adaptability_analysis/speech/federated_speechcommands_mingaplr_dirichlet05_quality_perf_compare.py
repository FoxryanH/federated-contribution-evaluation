import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SIBLING_SPEECH_DIR = REPO_ROOT / "1.accuracy_analysis" / "speechcommands"
for extra_path in (REPO_ROOT, SIBLING_SPEECH_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

import federated_speechcommands_hist_mingaplr_dirichlet_fixed_noise_common as dirichlet05_base
import federated_speechcommands_smartfl_mingaplr_compare as base


CODE_VERSION = "speechcommands_mingaplr_dirichlet05_quality_perf_compare_v1"
DIRICHLET_ALPHA = 0.5
DEFAULT_OUTPUT_JSON = Path("outputs/speechcommands_mingaplr_dirichlet05_quality_perf_compare/results.json")


def add_shared_args(parser, default_output_json):
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-json", type=Path, default=default_output_json)
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
    parser.add_argument("--server-lr-cap", type=float, default=3.0)
    parser.add_argument("--server-opt-steps", type=int, default=10)
    parser.add_argument("--server-opt-weight-tol", type=float, default=1e-4)
    parser.add_argument("--grad-damping", type=float, default=base.DEFAULT_GRAD_DAMPING)
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-duration-ms", type=int, default=1000)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--num-clean-clients", type=int, default=20)
    parser.add_argument("--num-noise20-clients", type=int, default=10)
    parser.add_argument("--num-noise40-clients", type=int, default=10)
    parser.add_argument("--num-noise60-clients", type=int, default=10)
    return parser


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Speech Commands performance comparison under Dirichlet-0.5 non-IID and quality heterogeneity: "
            "pure FedAvg vs FedAvg + server-side weight optimization"
        )
    )
    add_shared_args(parser, DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def validate_args(args):
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be <= num_clients")
    if args.proxy_batches_per_round <= 0:
        raise ValueError("proxy_batches_per_round must be positive")
    if args.task_size <= 0 or args.gen_test_size <= 0:
        raise ValueError("task_size and gen_test_size must be positive")

    expected_num_clients = (
        args.num_clean_clients
        + args.num_noise20_clients
        + args.num_noise40_clients
        + args.num_noise60_clients
    )
    if expected_num_clients != args.num_clients:
        raise ValueError(
            "Noise-group client counts must sum to num_clients: "
            f"{expected_num_clients} != {args.num_clients}"
        )


def build_client_noise_rates(args):
    noise_rates = []
    noise_rates.extend([0.0] * args.num_clean_clients)
    noise_rates.extend([0.2] * args.num_noise20_clients)
    noise_rates.extend([0.4] * args.num_noise40_clients)
    noise_rates.extend([0.6] * args.num_noise60_clients)
    return noise_rates


def build_client_noise_groups(client_noise_rates):
    group_labels = []
    for noise_rate in client_noise_rates:
        if noise_rate <= 0.0:
            group_labels.append("clean")
        elif abs(noise_rate - 0.2) < 1e-12:
            group_labels.append("noise20")
        elif abs(noise_rate - 0.4) < 1e-12:
            group_labels.append("noise40")
        elif abs(noise_rate - 0.6) < 1e-12:
            group_labels.append("noise60")
        else:
            group_labels.append(f"noise{int(round(noise_rate * 100))}")
    return group_labels


def build_fixed_noise_client_loaders(
    dataset,
    full_targets,
    client_indices,
    client_noise_rates,
    batch_size,
    loader_kwargs,
    base_seed,
):
    client_loaders = []
    client_sizes = []
    full_targets = np.asarray(full_targets, dtype=np.int64)

    for client_id, indices in enumerate(client_indices):
        noisy_labels = dirichlet05_base.build_fixed_noisy_labels(
            full_targets[np.asarray(indices, dtype=np.int64)],
            client_noise_rates[client_id],
            base_seed + client_id,
        )
        subset = dirichlet05_base.FixedLabelSubset(dataset, indices, noisy_labels)
        generator = torch.Generator()
        generator.manual_seed(base_seed + client_id)
        client_loader_kwargs = dict(loader_kwargs)
        client_loader_kwargs["generator"] = generator
        client_loaders.append(base.smartfl_base.DataLoader(subset, batch_size=batch_size, **client_loader_kwargs))
        client_sizes.append(len(indices))

    return client_loaders, client_sizes


def build_shared_experiment_state(args):
    base.set_seed(base.FIXED_SEED)
    devices = base.smartfl_base.resolve_devices(args)
    device = devices[0]

    full_dataset, full_targets = base.load_speech_commands(args)
    private_indices, task_indices, gen_test_indices = base.smartfl_base.stratified_three_way_split_indices(
        full_targets,
        args.task_size,
        args.gen_test_size,
        base.FIXED_SEED,
    )

    private_targets = full_targets[private_indices]
    client_indices, unused_private_indices, per_client_private_size = dirichlet05_base.equal_size_dirichlet_partition(
        private_indices,
        private_targets,
        args.num_clients,
        DIRICHLET_ALPHA,
        base.FIXED_SEED,
    )

    client_noise_rates = build_client_noise_rates(args)
    client_noise_groups = build_client_noise_groups(client_noise_rates)

    schedule_rng = random.Random(base.FIXED_SEED)
    selected_clients_schedule = [
        schedule_rng.sample(range(args.num_clients), args.clients_per_round)
        for _ in range(args.rounds)
    ]

    base.set_seed(base.FIXED_SEED)
    initial_model = base.build_model().to(device)
    initial_global_state = base.smartfl_base.clone_state_dict_to_cpu(initial_model.state_dict())

    return {
        "device": device,
        "full_dataset": full_dataset,
        "full_targets": full_targets,
        "private_indices": private_indices,
        "task_indices": task_indices,
        "gen_test_indices": gen_test_indices,
        "client_indices": client_indices,
        "client_noise_rates": client_noise_rates,
        "client_noise_groups": client_noise_groups,
        "selected_clients_schedule": selected_clients_schedule,
        "unused_private_indices": unused_private_indices,
        "per_client_private_size": per_client_private_size,
        "initial_global_state": initial_global_state,
    }


def build_run_loaders(args, shared_state):
    train_loader_kwargs = base.smartfl_base.build_loader_kwargs(args, shuffle=True)
    eval_loader_kwargs = base.smartfl_base.build_loader_kwargs(args, shuffle=False)

    client_loader_kwargs = dict(train_loader_kwargs)
    client_loader_kwargs["num_workers"] = 0
    client_loader_kwargs.pop("persistent_workers", None)

    client_loaders, client_sizes = build_fixed_noise_client_loaders(
        shared_state["full_dataset"],
        shared_state["full_targets"],
        shared_state["client_indices"],
        shared_state["client_noise_rates"],
        args.batch_size,
        client_loader_kwargs,
        base.FIXED_SEED,
    )

    task_subset = Subset(shared_state["full_dataset"], shared_state["task_indices"].tolist())
    proxy_batch_size = min(args.proxy_batch_size, len(task_subset))
    task_loader = DataLoader(task_subset, batch_size=proxy_batch_size, **train_loader_kwargs)
    task_iterator = iter(task_loader)
    task_eval_loader = DataLoader(task_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    gen_test_subset = Subset(shared_state["full_dataset"], shared_state["gen_test_indices"].tolist())
    gen_test_loader = DataLoader(gen_test_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    return {
        "client_loaders": client_loaders,
        "client_sizes": client_sizes,
        "task_loader": task_loader,
        "task_iterator": task_iterator,
        "task_eval_loader": task_eval_loader,
        "gen_test_loader": gen_test_loader,
        "proxy_batch_size": proxy_batch_size,
    }


def evaluate_global_model(global_model, task_eval_loader, gen_test_loader, device):
    task_loss, task_acc = base.smartfl_base.evaluate(global_model, task_eval_loader, device)
    gen_loss, gen_acc = base.smartfl_base.evaluate(global_model, gen_test_loader, device)
    return {
        "task_loss": float(task_loss),
        "task_acc": float(task_acc),
        "gen_loss": float(gen_loss),
        "gen_acc": float(gen_acc),
    }


def run_pure_fedavg(args, shared_state):
    device = shared_state["device"]
    loaders = build_run_loaders(args, shared_state)

    global_model = base.build_model().to(device)
    global_model.load_state_dict(shared_state["initial_global_state"])

    round_records = []

    for round_idx, selected_clients in enumerate(shared_state["selected_clients_schedule"], start=1):
        print(f"[FedAvg] round {round_idx:03d} selected_clients={selected_clients}")
        global_snapshot = base.smartfl_base.clone_state_dict_to_cpu(global_model.state_dict())

        local_states = []
        sample_counts = []
        for client_id in selected_clients:
            local_seed = base.FIXED_SEED + round_idx * 1000 + client_id
            local_state, sample_count = base.client_local_train(
                global_snapshot,
                loaders["client_loaders"][client_id],
                loaders["client_sizes"][client_id],
                device,
                args,
                local_seed,
            )
            local_states.append(local_state)
            sample_counts.append(sample_count)

        total_samples = float(sum(sample_counts))
        aggregation_weights = [sample_count / total_samples for sample_count in sample_counts]
        next_state = base.smartfl_base.aggregate_states_on_cpu(local_states, aggregation_weights, global_model)
        global_model.load_state_dict(next_state)

        metrics = evaluate_global_model(
            global_model,
            loaders["task_eval_loader"],
            loaders["gen_test_loader"],
            device,
        )
        round_records.append({"round_index": round_idx, **metrics})
        print(
            f"[FedAvg] round {round_idx:03d} done: "
            f"task_loss={metrics['task_loss']:.4f} "
            f"task_acc={metrics['task_acc']:.4f} "
            f"gen_loss={metrics['gen_loss']:.4f} "
            f"gen_acc={metrics['gen_acc']:.4f}"
        )

    return round_records


def run_ours(args, shared_state):
    device = shared_state["device"]
    loaders = build_run_loaders(args, shared_state)

    global_model = base.build_model().to(device)
    global_model.load_state_dict(shared_state["initial_global_state"])
    task_iterator = loaders["task_iterator"]

    round_records = []

    for round_idx, selected_clients in enumerate(shared_state["selected_clients_schedule"], start=1):
        print(f"[Ours] round {round_idx:03d} selected_clients={selected_clients}")
        proxy_batches, task_iterator = base.smartfl_base.next_batches(
            loaders["task_loader"],
            task_iterator,
            args.proxy_batches_per_round,
        )
        global_snapshot = base.smartfl_base.clone_state_dict_to_cpu(global_model.state_dict())

        local_states = []
        sample_counts = []
        for client_id in selected_clients:
            local_seed = base.FIXED_SEED + round_idx * 1000 + client_id
            local_state, sample_count = base.client_local_train(
                global_snapshot,
                loaders["client_loaders"][client_id],
                loaders["client_sizes"][client_id],
                device,
                args,
                local_seed,
            )
            local_states.append(local_state)
            sample_counts.append(sample_count)

        smartfl_result = base.smartfl_base.optimize_smartfl_weights(
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
        )
        next_state = base.smartfl_base.aggregate_states_on_cpu(local_states, smartfl_result["alpha"], global_model)
        global_model.load_state_dict(next_state)

        metrics = evaluate_global_model(
            global_model,
            loaders["task_eval_loader"],
            loaders["gen_test_loader"],
            device,
        )
        round_records.append({"round_index": round_idx, **metrics})
        print(
            f"[Ours] round {round_idx:03d} done: "
            f"task_loss={metrics['task_loss']:.4f} "
            f"task_acc={metrics['task_acc']:.4f} "
            f"gen_loss={metrics['gen_loss']:.4f} "
            f"gen_acc={metrics['gen_acc']:.4f}"
        )

    return round_records


def build_comparison_summary(fedavg_rounds, ours_rounds):
    if not fedavg_rounds or not ours_rounds:
        return {}

    fedavg_final = fedavg_rounds[-1]
    ours_final = ours_rounds[-1]

    return {
        "final_task_loss_fedavg": float(fedavg_final["task_loss"]),
        "final_task_loss_ours": float(ours_final["task_loss"]),
        "final_task_acc_fedavg": float(fedavg_final["task_acc"]),
        "final_task_acc_ours": float(ours_final["task_acc"]),
        "final_gen_loss_fedavg": float(fedavg_final["gen_loss"]),
        "final_gen_loss_ours": float(ours_final["gen_loss"]),
        "final_gen_acc_fedavg": float(fedavg_final["gen_acc"]),
        "final_gen_acc_ours": float(ours_final["gen_acc"]),
        "final_task_loss_drop": float(fedavg_final["task_loss"] - ours_final["task_loss"]),
        "final_task_acc_gain": float(ours_final["task_acc"] - fedavg_final["task_acc"]),
        "final_gen_loss_drop": float(fedavg_final["gen_loss"] - ours_final["gen_loss"]),
        "final_gen_acc_gain": float(ours_final["gen_acc"] - fedavg_final["gen_acc"]),
        "best_task_acc_fedavg": float(max(item["task_acc"] for item in fedavg_rounds)),
        "best_task_acc_ours": float(max(item["task_acc"] for item in ours_rounds)),
        "best_gen_acc_fedavg": float(max(item["gen_acc"] for item in fedavg_rounds)),
        "best_gen_acc_ours": float(max(item["gen_acc"] for item in ours_rounds)),
    }


def run_experiment(args):
    validate_args(args)
    shared_state = build_shared_experiment_state(args)

    print(
        "Quality-heterogeneous performance compare setup:",
        f"code_version={CODE_VERSION}",
        f"base_code_version={base.CODE_VERSION}",
        "dataset=SpeechCommands",
        "model=AudioCNN",
        "partition=Dirichlet",
        f"dirichlet_alpha={DIRICHLET_ALPHA}",
        "quality_heterogeneity=label_noise_only",
        "same_partition_same_schedule_same_init=true",
        f"clients={args.num_clients}",
        f"clients_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"private_train_size={len(shared_state['private_indices'])}",
        f"per_client_private_size={shared_state['per_client_private_size']}",
        f"unused_private_samples={len(shared_state['unused_private_indices'])}",
        f"task_size={len(shared_state['task_indices'])}",
        f"gen_test_size={len(shared_state['gen_test_indices'])}",
        f"proxy_batch_size={args.proxy_batch_size}",
        f"proxy_batches_per_round={args.proxy_batches_per_round}",
        f"noise_layout=clean:{args.num_clean_clients},noise20:{args.num_noise20_clients},"
        f"noise40:{args.num_noise40_clients},noise60:{args.num_noise60_clients}",
        f"labels={base.audio_base.COMMAND_LABELS}",
        f"fixed_seed={base.FIXED_SEED}",
        f"device={shared_state['device']}",
    )

    fedavg_rounds = run_pure_fedavg(args, shared_state)
    ours_rounds = run_ours(args, shared_state)

    payload = {
        "config": {
            "code_version": CODE_VERSION,
            "base_code_version": base.CODE_VERSION,
            "dataset": "SpeechCommands",
            "model": "AudioCNN",
            "labels": base.audio_base.COMMAND_LABELS,
            "partition": "Dirichlet",
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "quality_heterogeneity": "label_noise_only",
            "same_partition_same_schedule_same_init": True,
            "num_clients": args.num_clients,
            "clients_per_round": args.clients_per_round,
            "rounds": args.rounds,
            "task_size": len(shared_state["task_indices"]),
            "gen_test_size": len(shared_state["gen_test_indices"]),
            "private_train_size": len(shared_state["private_indices"]),
            "per_client_private_size": shared_state["per_client_private_size"],
            "unused_private_samples": len(shared_state["unused_private_indices"]),
            "proxy_batch_size": args.proxy_batch_size,
            "proxy_batches_per_round": args.proxy_batches_per_round,
            "fixed_seed": base.FIXED_SEED,
            "noise_layout": {
                "clean": args.num_clean_clients,
                "noise20": args.num_noise20_clients,
                "noise40": args.num_noise40_clients,
                "noise60": args.num_noise60_clients,
            },
            "methods": {
                "fedavg": "sample_count_weighted_fedavg",
                "ours": "fedavg_plus_server_side_quality_aware_weight_optimization",
            },
        },
        "fedavg_rounds": fedavg_rounds,
        "ours_rounds": ours_rounds,
        "comparison_summary": build_comparison_summary(fedavg_rounds, ours_rounds),
    }

    output_path = base.smartfl_base.resolve_output_path(args.output_json)
    base.smartfl_base.write_json_snapshot(payload, output_path)
    print(f"Speech Commands quality performance compare JSON saved to: {output_path}")


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
