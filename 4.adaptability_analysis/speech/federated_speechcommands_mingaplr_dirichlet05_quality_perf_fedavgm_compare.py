import argparse
from collections import OrderedDict
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SIBLING_SPEECH_DIR = REPO_ROOT / "1.accuracy_analysis" / "speechcommands"
for extra_path in (REPO_ROOT, SIBLING_SPEECH_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

import federated_speechcommands_smartfl_mingaplr_compare as base
import federated_speechcommands_mingaplr_dirichlet05_quality_perf_compare as perf_base


CODE_VERSION = "speechcommands_mingaplr_dirichlet05_quality_perf_fedavgm_compare_v1"
DEFAULT_OUTPUT_JSON = Path("outputs/speechcommands_mingaplr_dirichlet05_quality_perf_fedavgm_compare/results.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Speech Commands performance comparison under Dirichlet-0.5 non-IID and quality heterogeneity: "
            "pure FedAvgM vs FedAvgM + server-side weight optimization"
        )
    )
    perf_base.add_shared_args(parser, DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--fedavgm-server-momentum",
        type=float,
        default=0.5,
        help="Server momentum coefficient beta used by FedAvgM",
    )
    parser.add_argument(
        "--fedavgm-server-lr",
        type=float,
        default=1,
        help="Server learning rate eta used by FedAvgM",
    )
    return parser.parse_args()


def validate_args(args):
    perf_base.validate_args(args)
    if args.fedavgm_server_lr < 0:
        raise ValueError("fedavgm_server_lr must be non-negative")
    if args.fedavgm_server_momentum < 0:
        raise ValueError("fedavgm_server_momentum must be non-negative")


def build_fedavgm_momentum_state(model):
    param_names = set(base.smartfl_base.get_param_names(model))
    momentum_state = OrderedDict()

    for key, value in model.state_dict().items():
        if key in param_names:
            momentum_state[key] = torch.zeros_like(value.detach(), device="cpu")

    return momentum_state


def assert_state_is_finite(state_dict, context):
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite tensor detected in {context}: key={key}")


def apply_fedavgm_update(global_state, aggregated_state, momentum_state, model, server_lr, server_momentum):
    param_names = set(base.smartfl_base.get_param_names(model))
    next_state = OrderedDict()
    next_momentum_state = OrderedDict()

    for key, global_value in global_state.items():
        aggregated_value = aggregated_state[key]
        if key in param_names:
            delta = (aggregated_value - global_value).detach().clone().cpu()
            previous_momentum = momentum_state[key]
            updated_momentum = (server_momentum * previous_momentum + delta).detach().clone().cpu()
            next_momentum_state[key] = updated_momentum
            next_state[key] = (global_value + server_lr * updated_momentum).detach().clone().cpu()
        elif torch.is_tensor(aggregated_value):
            next_state[key] = aggregated_value.detach().clone().cpu()
        else:
            next_state[key] = aggregated_value

    assert_state_is_finite(next_state, "fedavgm_next_state")
    return next_state, next_momentum_state


def build_comparison_summary(fedavgm_rounds, ours_rounds):
    if not fedavgm_rounds or not ours_rounds:
        return {}

    fedavgm_final = fedavgm_rounds[-1]
    ours_final = ours_rounds[-1]

    return {
        "final_task_loss_fedavgm": float(fedavgm_final["task_loss"]),
        "final_task_loss_ours": float(ours_final["task_loss"]),
        "final_task_acc_fedavgm": float(fedavgm_final["task_acc"]),
        "final_task_acc_ours": float(ours_final["task_acc"]),
        "final_gen_loss_fedavgm": float(fedavgm_final["gen_loss"]),
        "final_gen_loss_ours": float(ours_final["gen_loss"]),
        "final_gen_acc_fedavgm": float(fedavgm_final["gen_acc"]),
        "final_gen_acc_ours": float(ours_final["gen_acc"]),
        "final_task_loss_drop": float(fedavgm_final["task_loss"] - ours_final["task_loss"]),
        "final_task_acc_gain": float(ours_final["task_acc"] - fedavgm_final["task_acc"]),
        "final_gen_loss_drop": float(fedavgm_final["gen_loss"] - ours_final["gen_loss"]),
        "final_gen_acc_gain": float(ours_final["gen_acc"] - fedavgm_final["gen_acc"]),
        "best_task_acc_fedavgm": float(max(item["task_acc"] for item in fedavgm_rounds)),
        "best_task_acc_ours": float(max(item["task_acc"] for item in ours_rounds)),
        "best_gen_acc_fedavgm": float(max(item["gen_acc"] for item in fedavgm_rounds)),
        "best_gen_acc_ours": float(max(item["gen_acc"] for item in ours_rounds)),
    }


def run_pure_fedavgm(args, shared_state):
    device = shared_state["device"]
    loaders = perf_base.build_run_loaders(args, shared_state)

    global_model = base.build_model().to(device)
    global_model.load_state_dict(shared_state["initial_global_state"])
    server_momentum_state = build_fedavgm_momentum_state(global_model)

    round_records = []

    for round_idx, selected_clients in enumerate(shared_state["selected_clients_schedule"], start=1):
        print(f"[FedAvgM] round {round_idx:03d} selected_clients={selected_clients}")
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
        aggregated_state = base.smartfl_base.aggregate_states_on_cpu(local_states, aggregation_weights, global_model)
        next_state, server_momentum_state = apply_fedavgm_update(
            global_snapshot,
            aggregated_state,
            server_momentum_state,
            global_model,
            args.fedavgm_server_lr,
            args.fedavgm_server_momentum,
        )
        global_model.load_state_dict(next_state)

        metrics = perf_base.evaluate_global_model(
            global_model,
            loaders["task_eval_loader"],
            loaders["gen_test_loader"],
            device,
        )
        round_records.append({"round_index": round_idx, **metrics})
        print(
            f"[FedAvgM] round {round_idx:03d} done: "
            f"task_loss={metrics['task_loss']:.4f} "
            f"task_acc={metrics['task_acc']:.4f} "
            f"gen_loss={metrics['gen_loss']:.4f} "
            f"gen_acc={metrics['gen_acc']:.4f}"
        )

    return round_records


def run_ours_fedavgm(args, shared_state):
    device = shared_state["device"]
    loaders = perf_base.build_run_loaders(args, shared_state)

    global_model = base.build_model().to(device)
    global_model.load_state_dict(shared_state["initial_global_state"])
    server_momentum_state = build_fedavgm_momentum_state(global_model)
    task_iterator = loaders["task_iterator"]

    round_records = []

    for round_idx, selected_clients in enumerate(shared_state["selected_clients_schedule"], start=1):
        print(f"[FedAvgM + Ours] round {round_idx:03d} selected_clients={selected_clients}")
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
        aggregated_state = base.smartfl_base.aggregate_states_on_cpu(local_states, smartfl_result["alpha"], global_model)
        next_state, server_momentum_state = apply_fedavgm_update(
            global_snapshot,
            aggregated_state,
            server_momentum_state,
            global_model,
            args.fedavgm_server_lr,
            args.fedavgm_server_momentum,
        )
        global_model.load_state_dict(next_state)

        metrics = perf_base.evaluate_global_model(
            global_model,
            loaders["task_eval_loader"],
            loaders["gen_test_loader"],
            device,
        )
        round_records.append({"round_index": round_idx, **metrics})
        print(
            f"[FedAvgM + Ours] round {round_idx:03d} done: "
            f"task_loss={metrics['task_loss']:.4f} "
            f"task_acc={metrics['task_acc']:.4f} "
            f"gen_loss={metrics['gen_loss']:.4f} "
            f"gen_acc={metrics['gen_acc']:.4f}"
        )

    return round_records


def run_experiment(args):
    validate_args(args)
    shared_state = perf_base.build_shared_experiment_state(args)

    print(
        "Quality-heterogeneous FedAvgM performance compare setup:",
        f"code_version={CODE_VERSION}",
        f"base_code_version={base.CODE_VERSION}",
        f"reference_perf_code_version={perf_base.CODE_VERSION}",
        "dataset=SpeechCommands",
        "model=AudioCNN",
        "partition=Dirichlet",
        f"dirichlet_alpha={perf_base.DIRICHLET_ALPHA}",
        "quality_heterogeneity=label_noise_only",
        "same_partition_same_schedule_same_init=true",
        f"clients={args.num_clients}",
        f"clients_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"fedavgm_server_momentum={args.fedavgm_server_momentum}",
        f"fedavgm_server_lr={args.fedavgm_server_lr}",
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

    fedavgm_rounds = run_pure_fedavgm(args, shared_state)
    ours_rounds = run_ours_fedavgm(args, shared_state)

    payload = {
        "config": {
            "code_version": CODE_VERSION,
            "base_code_version": base.CODE_VERSION,
            "reference_perf_code_version": perf_base.CODE_VERSION,
            "dataset": "SpeechCommands",
            "model": "AudioCNN",
            "labels": base.audio_base.COMMAND_LABELS,
            "partition": "Dirichlet",
            "dirichlet_alpha": perf_base.DIRICHLET_ALPHA,
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
            "fedavgm_server_momentum": args.fedavgm_server_momentum,
            "fedavgm_server_lr": args.fedavgm_server_lr,
            "fixed_seed": base.FIXED_SEED,
            "noise_layout": {
                "clean": args.num_clean_clients,
                "noise20": args.num_noise20_clients,
                "noise40": args.num_noise40_clients,
                "noise60": args.num_noise60_clients,
            },
            "methods": {
                "fedavgm": "sample_count_weighted_fedavgm",
                "ours": "fedavgm_plus_server_side_quality_aware_weight_optimization",
            },
        },
        "fedavgm_rounds": fedavgm_rounds,
        "ours_rounds": ours_rounds,
        "comparison_summary": build_comparison_summary(fedavgm_rounds, ours_rounds),
    }

    output_path = base.smartfl_base.resolve_output_path(args.output_json)
    base.smartfl_base.write_json_snapshot(payload, output_path)
    print(f"Speech Commands FedAvgM quality performance compare JSON saved to: {output_path}")


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
