import argparse
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SIBLING_CIFAR10_DIR = REPO_ROOT / "1.accuracy_analysis" / "cifar10"
for extra_path in (REPO_ROOT, SIBLING_CIFAR10_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

import federated_cifar10_mingaplr_compare as base
import federated_cifar10_mingaplr_dirichlet05_quality_perf_compare as perf_base


CODE_VERSION = "mingaplr_dirichlet05_quality_perf_fedprox_compare_v1"
DEFAULT_OUTPUT_JSON = Path("outputs/mingaplr_dirichlet05_quality_perf_fedprox_compare/results.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CIFAR-10 performance comparison under Dirichlet-0.5 non-IID and quality heterogeneity: "
            "pure FedProx vs FedProx + server-side weight optimization"
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
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
        "--fedprox-mu",
        type=float,
        default=0.02,
        help="FedProx proximal coefficient mu",
    )
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-clean-clients", type=int, default=20)
    parser.add_argument("--num-noise20-clients", type=int, default=10)
    parser.add_argument("--num-noise40-clients", type=int, default=10)
    parser.add_argument("--num-noise60-clients", type=int, default=10)
    return parser.parse_args()


def client_local_train_fedprox(global_state, dataset_loader, sample_count, device, args, local_seed):
    local_model = base.build_model().to(device)
    local_model.load_state_dict(global_state)
    local_model.train()

    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)

    optimizer = base.build_optimizer(local_model, args)
    criterion = torch.nn.CrossEntropyLoss()
    global_param_refs = [param.detach().clone() for param in local_model.parameters()]

    for _ in range(args.local_epochs):
        for inputs, targets in dataset_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, targets)

            if args.fedprox_mu > 0:
                prox_term = torch.zeros((), device=device)
                for param, ref_param in zip(local_model.parameters(), global_param_refs):
                    prox_term = prox_term + torch.sum((param - ref_param) ** 2)
                loss = loss + 0.5 * args.fedprox_mu * prox_term

            loss.backward()
            optimizer.step()

    local_state = base.clone_state_dict_to_cpu(local_model.state_dict())
    del optimizer
    del criterion
    del local_model
    base.clear_device_cache(device)
    return local_state, sample_count


def build_comparison_summary(fedprox_rounds, ours_rounds):
    if not fedprox_rounds or not ours_rounds:
        return {}

    fedprox_final = fedprox_rounds[-1]
    ours_final = ours_rounds[-1]

    return {
        "final_task_loss_fedprox": float(fedprox_final["task_loss"]),
        "final_task_loss_ours": float(ours_final["task_loss"]),
        "final_task_acc_fedprox": float(fedprox_final["task_acc"]),
        "final_task_acc_ours": float(ours_final["task_acc"]),
        "final_gen_loss_fedprox": float(fedprox_final["gen_loss"]),
        "final_gen_loss_ours": float(ours_final["gen_loss"]),
        "final_gen_acc_fedprox": float(fedprox_final["gen_acc"]),
        "final_gen_acc_ours": float(ours_final["gen_acc"]),
        "final_task_loss_drop": float(fedprox_final["task_loss"] - ours_final["task_loss"]),
        "final_task_acc_gain": float(ours_final["task_acc"] - fedprox_final["task_acc"]),
        "final_gen_loss_drop": float(fedprox_final["gen_loss"] - ours_final["gen_loss"]),
        "final_gen_acc_gain": float(ours_final["gen_acc"] - fedprox_final["gen_acc"]),
        "best_task_acc_fedprox": float(max(item["task_acc"] for item in fedprox_rounds)),
        "best_task_acc_ours": float(max(item["task_acc"] for item in ours_rounds)),
        "best_gen_acc_fedprox": float(max(item["gen_acc"] for item in fedprox_rounds)),
        "best_gen_acc_ours": float(max(item["gen_acc"] for item in ours_rounds)),
    }


def run_pure_fedprox(args, shared_state):
    device = shared_state["device"]
    loaders = perf_base.build_run_loaders(args, shared_state)

    global_model = base.build_model().to(device)
    global_model.load_state_dict(shared_state["initial_global_state"])

    round_records = []

    for round_idx, selected_clients in enumerate(shared_state["selected_clients_schedule"], start=1):
        print(f"[FedProx] round {round_idx:03d} selected_clients={selected_clients}")
        global_snapshot = base.clone_state_dict_to_cpu(global_model.state_dict())

        local_states = []
        sample_counts = []
        for client_id in selected_clients:
            local_seed = base.FIXED_SEED + round_idx * 1000 + client_id
            local_state, sample_count = client_local_train_fedprox(
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
        next_state = base.aggregate_states_on_cpu(local_states, aggregation_weights, global_model)
        global_model.load_state_dict(next_state)

        metrics = perf_base.evaluate_global_model(
            global_model,
            loaders["task_eval_loader"],
            loaders["gen_test_loader"],
            device,
        )
        round_records.append({"round_index": round_idx, **metrics})
        print(
            f"[FedProx] round {round_idx:03d} done: "
            f"task_loss={metrics['task_loss']:.4f} "
            f"task_acc={metrics['task_acc']:.4f} "
            f"gen_loss={metrics['gen_loss']:.4f} "
            f"gen_acc={metrics['gen_acc']:.4f}"
        )

    return round_records


def run_ours_fedprox(args, shared_state):
    device = shared_state["device"]
    loaders = perf_base.build_run_loaders(args, shared_state)

    global_model = base.build_model().to(device)
    global_model.load_state_dict(shared_state["initial_global_state"])
    task_iterator = loaders["task_iterator"]

    round_records = []

    for round_idx, selected_clients in enumerate(shared_state["selected_clients_schedule"], start=1):
        print(f"[FedProx + Ours] round {round_idx:03d} selected_clients={selected_clients}")
        proxy_batches, task_iterator = base.next_batches(
            loaders["task_loader"],
            task_iterator,
            args.proxy_batches_per_round,
        )
        global_snapshot = base.clone_state_dict_to_cpu(global_model.state_dict())

        local_states = []
        sample_counts = []
        for client_id in selected_clients:
            local_seed = base.FIXED_SEED + round_idx * 1000 + client_id
            local_state, sample_count = client_local_train_fedprox(
                global_snapshot,
                loaders["client_loaders"][client_id],
                loaders["client_sizes"][client_id],
                device,
                args,
                local_seed,
            )
            local_states.append(local_state)
            sample_counts.append(sample_count)

        smartfl_result = base.optimize_smartfl_weights(
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
        next_state = base.aggregate_states_on_cpu(local_states, smartfl_result["alpha"], global_model)
        global_model.load_state_dict(next_state)

        metrics = perf_base.evaluate_global_model(
            global_model,
            loaders["task_eval_loader"],
            loaders["gen_test_loader"],
            device,
        )
        round_records.append({"round_index": round_idx, **metrics})
        print(
            f"[FedProx + Ours] round {round_idx:03d} done: "
            f"task_loss={metrics['task_loss']:.4f} "
            f"task_acc={metrics['task_acc']:.4f} "
            f"gen_loss={metrics['gen_loss']:.4f} "
            f"gen_acc={metrics['gen_acc']:.4f}"
        )

    return round_records


def run_experiment(args):
    perf_base.validate_args(args)
    shared_state = perf_base.build_shared_experiment_state(args)

    print(
        "Quality-heterogeneous FedProx performance compare setup:",
        f"code_version={CODE_VERSION}",
        f"base_code_version={base.CODE_VERSION}",
        "dataset=CIFAR-10",
        "partition=Dirichlet",
        f"dirichlet_alpha={perf_base.DIRICHLET_ALPHA}",
        "quality_heterogeneity=label_noise_only",
        "same_partition_same_schedule_same_init=true",
        f"clients={args.num_clients}",
        f"clients_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"fedprox_mu={args.fedprox_mu}",
        f"private_train_size={len(shared_state['private_indices'])}",
        f"per_client_private_size={shared_state['per_client_private_size']}",
        f"unused_private_samples={len(shared_state['unused_private_indices'])}",
        f"task_size={len(shared_state['task_indices'])}",
        f"gen_test_size={len(shared_state['gen_test_indices'])}",
        f"proxy_batch_size={args.proxy_batch_size}",
        f"proxy_batches_per_round={args.proxy_batches_per_round}",
        f"noise_layout=clean:{args.num_clean_clients},noise20:{args.num_noise20_clients},"
        f"noise40:{args.num_noise40_clients},noise60:{args.num_noise60_clients}",
        f"fixed_seed={base.FIXED_SEED}",
        f"device={shared_state['device']}",
    )

    fedprox_rounds = run_pure_fedprox(args, shared_state)
    ours_rounds = run_ours_fedprox(args, shared_state)

    payload = {
        "config": {
            "code_version": CODE_VERSION,
            "base_code_version": base.CODE_VERSION,
            "reference_perf_code_version": perf_base.CODE_VERSION,
            "dataset": "CIFAR-10",
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
            "fedprox_mu": args.fedprox_mu,
            "fixed_seed": base.FIXED_SEED,
            "noise_layout": {
                "clean": args.num_clean_clients,
                "noise20": args.num_noise20_clients,
                "noise40": args.num_noise40_clients,
                "noise60": args.num_noise60_clients,
            },
            "methods": {
                "fedprox": "sample_count_weighted_fedprox",
                "ours": "fedprox_plus_server_side_quality_aware_weight_optimization",
            },
        },
        "fedprox_rounds": fedprox_rounds,
        "ours_rounds": ours_rounds,
        "comparison_summary": build_comparison_summary(fedprox_rounds, ours_rounds),
    }

    output_path = base.resolve_output_path(args.output_json)
    base.write_json_snapshot(payload, output_path)
    print(f"Quality FedProx performance compare JSON saved to: {output_path}")


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
