import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SIBLING_CIFAR10_DIR = REPO_ROOT / "1.accuracy_analysis" / "cifar10"
for extra_path in (REPO_ROOT, SIBLING_CIFAR10_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

import federated_cifar10_mingaplr_compare as base
import federated_cifar10_mingaplr_S1_controlled_compare as controlled_base


CODE_VERSION = "loo_controlled_compare_s1_global_v2_adaptive_shrinkage"
DEFAULT_OUTPUT_JSON = Path("outputs/loo_controlled_compare/s1_global_results.json")
DEFAULT_PROGRESS_LOG_EVERY = 1
EPS = 1e-8
PREFERRED_CATEGORY_ORDER = ["clean", "noise20", "noise40", "noise60", "iid_shared_distribution"]
UTILITY_METRIC_CHOICES = ["gen_acc", "task_acc", "neg_gen_loss", "neg_task_loss"]
CONTROLLED_FIXED_SEED = getattr(controlled_base, "CONTROLLED_FIXED_SEED", getattr(base, "FIXED_SEED", 42))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled CIFAR-10 global leave-one-out baseline with configurable client size, noise rate, and missing classes"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--scenario-tag", type=str, default="S1")
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=-1,
        help="Set to -1 to select all coalition members every round",
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--task-size", type=int, default=1000)
    parser.add_argument("--gen-test-size", type=int, default=10000)
    parser.add_argument("--local-optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--local-lr", type=float, default=0.01)
    parser.add_argument("--local-momentum", type=float, default=0.9)
    parser.add_argument("--local-weight-decay", type=float, default=0.0)
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--client-data-sizes",
        type=str,
        default="3000",
        help="Single value or comma-separated list of per-client sample counts",
    )
    parser.add_argument(
        "--client-noise-rates",
        type=str,
        default="0",
        help="Single value or comma-separated list of per-client noise rates; values >1 are treated as percentages",
    )
    parser.add_argument(
        "--client-missing-classes",
        type=str,
        default="0",
        help="Single value or comma-separated list of how many classes each client is missing",
    )
    parser.add_argument(
        "--loo-utility-metric",
        type=str,
        default="gen_acc",
        choices=UTILITY_METRIC_CHOICES,
    )
    parser.add_argument(
        "--verbose-round-logs",
        action="store_true",
        help="Print per-round progress inside each run",
    )
    parser.add_argument(
        "--progress-log-every",
        type=int,
        default=DEFAULT_PROGRESS_LOG_EVERY,
        help="Print a lightweight training progress line every N rounds inside each run",
    )
    parser.add_argument(
        "--print-missing-classes-on-round1",
        action="store_true",
        help="Print controlled missing-class settings before training starts",
    )
    parser.add_argument(
        "--loo-shrinkage-gamma",
        type=float,
        default=1.0,
        help="Adaptive shrinkage sensitivity gamma used in beta=D(q,u)^gamma",
    )
    args = parser.parse_args()
    if args.clients_per_round <= 0:
        args.clients_per_round = args.num_clients
    if args.progress_log_every <= 0:
        raise ValueError("progress-log-every must be a positive integer.")
    if args.loo_shrinkage_gamma <= 0:
        raise ValueError("loo-shrinkage-gamma must be positive.")
    return args


def print_loo_setup_banner(
    args,
    private_indices,
    task_indices,
    gen_test_indices,
    device,
    requested_client_samples,
    unused_private_samples,
):
    print(
        "Controlled global-LOO setup:",
        f"code_version={CODE_VERSION}",
        "dataset=CIFAR-10",
        f"scenario={args.scenario_tag}",
        "training_aggregation=FedAvg(sample_count_weighted)",
        "contribution_method=global_leave_one_out",
        f"loo_utility_metric={args.loo_utility_metric}",
        "loo_scope=whole_training_run",
        "normalized_contribution=adaptive_uniform_shrinkage_over_positive_loo",
        f"loo_shrinkage_gamma={args.loo_shrinkage_gamma}",
        f"clients={args.num_clients}",
        f"clients_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"client_train_total={requested_client_samples}",
        f"private_pool={len(private_indices)}",
        f"unused_private_pool={unused_private_samples}",
        f"eval_task_set={len(task_indices)}",
        f"eval_gen_test={len(gen_test_indices)}",
        f"run_count={args.num_clients + 1}",
        f"fixed_seed={CONTROLLED_FIXED_SEED}",
        f"device={device}",
    )


def build_category_average_contributions(client_contributions, contribution_key="average_normalized_contribution"):
    encountered_categories = []
    for item in client_contributions:
        category = item.get("client_category", "unknown")
        if category not in encountered_categories:
            encountered_categories.append(category)

    category_order = [category for category in PREFERRED_CATEGORY_ORDER if category in encountered_categories]
    category_order.extend(
        [category for category in encountered_categories if category not in PREFERRED_CATEGORY_ORDER]
    )

    results = []
    for category in category_order:
        values = [
            float(item[contribution_key])
            for item in client_contributions
            if item.get("client_category", "unknown") == category
            and item.get(contribution_key) is not None
        ]
        results.append(
            {
                "client_category": category,
                "average_normalized_contribution": (sum(values) / len(values)) if values else None,
            }
        )
    return results


def build_normalized_contributions(loo_values):
    positive_parts = [max(value, 0.0) for value in loo_values]
    positive_sum = sum(positive_parts)
    if positive_sum > EPS:
        return [value / positive_sum for value in positive_parts]
    if not loo_values:
        return []
    uniform_value = 1.0 / len(loo_values)
    return [uniform_value for _ in loo_values]


def compute_uniform_deviation(normalized_contributions):
    num_clients = len(normalized_contributions)
    if num_clients <= 1:
        return 0.0
    uniform_value = 1.0 / num_clients
    numerator = sum(abs(value - uniform_value) for value in normalized_contributions)
    denominator = 2.0 * (1.0 - uniform_value)
    if denominator <= EPS:
        return 0.0
    deviation = numerator / denominator
    return max(0.0, min(1.0, float(deviation)))


def compute_adaptive_shrinkage_beta(normalized_contributions, gamma):
    deviation = compute_uniform_deviation(normalized_contributions)
    beta = deviation ** gamma
    beta = max(0.0, min(1.0, float(beta)))
    return beta, deviation


def apply_uniform_shrinkage(normalized_contributions, beta):
    num_clients = len(normalized_contributions)
    if num_clients == 0:
        return []
    uniform_value = 1.0 / num_clients
    return [
        float((1.0 - beta) * uniform_value + beta * value)
        for value in normalized_contributions
    ]


def evaluate_for_utility_metric(model, utility_metric, task_eval_loader, gen_test_loader, device):
    if utility_metric in ("gen_acc", "neg_gen_loss"):
        gen_loss, gen_acc = base.evaluate(model, gen_test_loader, device)
        return {"gen_loss": gen_loss, "gen_acc": gen_acc}
    if utility_metric in ("task_acc", "neg_task_loss"):
        task_loss, task_acc = base.evaluate(model, task_eval_loader, device)
        return {"task_loss": task_loss, "task_acc": task_acc}
    raise ValueError(f"Unsupported loo utility metric: {utility_metric}")


def compute_absolute_utility(metrics, utility_metric):
    if utility_metric == "gen_acc":
        return float(metrics["gen_acc"])
    if utility_metric == "task_acc":
        return float(metrics["task_acc"])
    if utility_metric == "neg_gen_loss":
        return float(-metrics["gen_loss"])
    if utility_metric == "neg_task_loss":
        return float(-metrics["task_loss"])
    raise ValueError(f"Unsupported loo utility metric: {utility_metric}")


def format_metric_summary(metrics, utility_metric):
    if utility_metric == "gen_acc":
        return f"gen_acc={metrics['gen_acc']:.4f}"
    if utility_metric == "task_acc":
        return f"task_acc={metrics['task_acc']:.4f}"
    if utility_metric == "neg_gen_loss":
        return f"gen_loss={metrics['gen_loss']:.4f}"
    if utility_metric == "neg_task_loss":
        return f"task_loss={metrics['task_loss']:.4f}"
    raise ValueError(f"Unsupported loo utility metric: {utility_metric}")


def get_missing_classes_info(selected_clients, client_metadata):
    helper = getattr(controlled_base, "build_selected_clients_missing_classes_info", None)
    if helper is not None:
        return helper(selected_clients, client_metadata)
    tokens = []
    for client_id in selected_clients:
        missing_classes = client_metadata[client_id].get("missing_classes", [])
        rendered = ",".join(str(class_id) for class_id in missing_classes) if missing_classes else "none"
        tokens.append(f"p{client_id + 1}:[{rendered}]")
    return " | ".join(tokens)


def run_fedavg_training_for_client_set(
    coalition_clients,
    args,
    client_loaders,
    client_sizes,
    initial_global_state,
    task_eval_loader,
    gen_test_loader,
    device,
    run_label,
):
    coalition_clients = list(sorted(coalition_clients))
    coalition_model = base.build_model().to(device)
    coalition_model.load_state_dict(initial_global_state)

    if not coalition_clients:
        metrics = evaluate_for_utility_metric(
            coalition_model, args.loo_utility_metric, task_eval_loader, gen_test_loader, device
        )
        return {"coalition_clients": coalition_clients, **metrics}

    effective_clients_per_round = min(args.clients_per_round, len(coalition_clients))

    for round_idx in range(1, args.rounds + 1):
        if effective_clients_per_round == len(coalition_clients):
            selected_clients = list(coalition_clients)
        else:
            selected_clients = base.random.sample(coalition_clients, effective_clients_per_round)

        if args.verbose_round_logs:
            print(
                f"  {run_label} round {round_idx:03d}/{args.rounds:03d} "
                f"selected_clients={selected_clients}"
            )

        global_snapshot = base.clone_state_dict_to_cpu(coalition_model.state_dict())
        local_states = []
        sample_counts = []

        for client_id in selected_clients:
            local_seed = CONTROLLED_FIXED_SEED + round_idx * 1000 + client_id
            local_state, sample_count = base.client_local_train(
                global_snapshot,
                client_loaders[client_id],
                client_sizes[client_id],
                device,
                args,
                local_seed,
            )
            local_states.append(local_state)
            sample_counts.append(sample_count)

        total_samples = float(sum(sample_counts))
        aggregation_weights = [sample_count / total_samples for sample_count in sample_counts]
        next_state = base.aggregate_states_on_cpu(local_states, aggregation_weights, coalition_model)
        coalition_model.load_state_dict(next_state)

        if (
            round_idx % args.progress_log_every == 0
            or round_idx == 1
            or round_idx == args.rounds
        ):
            print(
                f"  {run_label} progress round {round_idx:03d}/{args.rounds:03d} "
                f"trained_clients={len(selected_clients)}"
            )

    metrics = evaluate_for_utility_metric(
        coalition_model, args.loo_utility_metric, task_eval_loader, gen_test_loader, device
    )
    return {"coalition_clients": coalition_clients, **metrics}


def build_client_loo_contributions(
    loo_values,
    raw_normalized_contributions,
    stabilized_contributions,
    client_metadata,
):
    client_contributions = []
    for client_id, loo_value in enumerate(loo_values):
        client_contributions.append(
            {
                "client_id": client_id,
                **client_metadata[client_id],
                "loo_value": float(loo_value),
                "loo_positive_part": float(max(loo_value, 0.0)),
                "raw_normalized_contribution": float(raw_normalized_contributions[client_id]),
                "average_normalized_contribution": float(stabilized_contributions[client_id]),
                "stabilized_normalized_contribution": float(stabilized_contributions[client_id]),
            }
        )
    return client_contributions


def run_loo_experiment(args):
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be <= num_clients")
    if args.loo_utility_metric not in UTILITY_METRIC_CHOICES:
        raise ValueError(
            f"Unsupported loo utility metric {args.loo_utility_metric}. "
            f"Choose from {UTILITY_METRIC_CHOICES}."
        )

    client_data_sizes = controlled_base.parse_per_client_values(
        args.client_data_sizes, args.num_clients, "client-data-sizes", controlled_base.cast_client_size
    )
    client_noise_rates = controlled_base.parse_per_client_values(
        args.client_noise_rates, args.num_clients, "client-noise-rates", controlled_base.cast_noise_rate
    )
    client_missing_classes = controlled_base.parse_per_client_values(
        args.client_missing_classes, args.num_clients, "client-missing-classes", controlled_base.cast_missing_classes
    )

    base.set_seed(CONTROLLED_FIXED_SEED)
    devices = base.resolve_devices(args)
    device = devices[0]

    train_dataset, test_dataset = base.load_cifar10(args.data_dir)
    full_dataset = base.ConcatDataset([train_dataset, test_dataset])
    full_targets = base.np.asarray(train_dataset.targets + test_dataset.targets, dtype=base.np.int64)
    private_indices, task_indices, gen_test_indices = base.stratified_three_way_split_indices(
        full_targets,
        args.task_size,
        args.gen_test_size,
        CONTROLLED_FIXED_SEED,
    )

    if sum(client_data_sizes) > len(private_indices):
        raise ValueError(
            f"Requested {sum(client_data_sizes)} total client samples, but only {len(private_indices)} private samples are available."
        )

    private_targets = full_targets[private_indices]
    (
        client_indices,
        client_available_classes,
        client_missing_class_lists,
        client_target_class_counts,
        unused_private_indices,
    ) = controlled_base.build_controlled_client_indices(
        private_indices,
        private_targets,
        client_data_sizes,
        client_missing_classes,
        CONTROLLED_FIXED_SEED,
    )
    client_configs = controlled_base.build_client_configs(
        client_data_sizes,
        client_noise_rates,
        client_available_classes,
        client_missing_classes,
        client_missing_class_lists,
    )

    train_loader_kwargs = base.build_loader_kwargs(args, shuffle=True)
    eval_loader_kwargs = base.build_loader_kwargs(args, shuffle=False)
    client_loader_kwargs = dict(train_loader_kwargs)
    client_loader_kwargs["num_workers"] = 0
    client_loader_kwargs.pop("persistent_workers", None)

    client_loaders, client_sizes = controlled_base.build_controlled_client_loaders(
        full_dataset,
        full_targets,
        client_indices,
        client_noise_rates,
        args.batch_size,
        client_loader_kwargs,
        CONTROLLED_FIXED_SEED,
    )

    task_subset = base.Subset(full_dataset, task_indices.tolist())
    task_eval_loader = base.DataLoader(task_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)
    gen_test_subset = base.Subset(full_dataset, gen_test_indices.tolist())
    gen_test_loader = base.DataLoader(gen_test_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    print_loo_setup_banner(
        args=args,
        private_indices=private_indices,
        task_indices=task_indices,
        gen_test_indices=gen_test_indices,
        device=device,
        requested_client_samples=sum(client_data_sizes),
        unused_private_samples=len(unused_private_indices),
    )
    print("Controlled clients:", controlled_base.format_client_configs_for_banner(client_configs))
    if getattr(args, "print_missing_classes_on_round1", False):
        print(
            "Controlled missing classes:",
            get_missing_classes_info(range(args.num_clients), client_configs),
        )

    base.set_seed(CONTROLLED_FIXED_SEED)
    initial_model = base.build_model().to(device)
    initial_global_state = base.clone_state_dict_to_cpu(initial_model.state_dict())

    all_clients = list(range(args.num_clients))
    run_count = args.num_clients + 1
    run_index = 1

    full_run_label = f"[run {run_index:02d}/{run_count:02d}]"
    print(f"{full_run_label} start clients={all_clients}")
    full_metrics = run_fedavg_training_for_client_set(
        coalition_clients=all_clients,
        args=args,
        client_loaders=client_loaders,
        client_sizes=client_sizes,
        initial_global_state=initial_global_state,
        task_eval_loader=task_eval_loader,
        gen_test_loader=gen_test_loader,
        device=device,
        run_label=full_run_label,
    )
    full_metrics["utility"] = compute_absolute_utility(full_metrics, args.loo_utility_metric)
    print(
        f"{full_run_label} finished: "
        f"{format_metric_summary(full_metrics, args.loo_utility_metric)} "
        f"utility={full_metrics['utility']:.6f}"
    )

    leave_one_out_results = []
    loo_values = [0.0 for _ in range(args.num_clients)]

    for client_id in range(args.num_clients):
        run_index += 1
        retained_clients = [idx for idx in all_clients if idx != client_id]
        run_label = f"[run {run_index:02d}/{run_count:02d}]"
        print(f"{run_label} start excluded_client={client_id} retained_clients={retained_clients}")
        metrics = run_fedavg_training_for_client_set(
            coalition_clients=retained_clients,
            args=args,
            client_loaders=client_loaders,
            client_sizes=client_sizes,
            initial_global_state=initial_global_state,
            task_eval_loader=task_eval_loader,
            gen_test_loader=gen_test_loader,
            device=device,
            run_label=run_label,
        )
        metrics["utility"] = compute_absolute_utility(metrics, args.loo_utility_metric)
        loo_value = float(full_metrics["utility"] - metrics["utility"])
        loo_values[client_id] = loo_value
        print(
            f"{run_label} finished: "
            f"{format_metric_summary(metrics, args.loo_utility_metric)} "
            f"utility={metrics['utility']:.6f} "
            f"loo_value={loo_value:.6f}"
        )
        leave_one_out_results.append(
            {
                "excluded_client_id": client_id,
                "excluded_participant_id": client_configs[client_id]["participant_id"],
                "retained_clients": retained_clients,
                **{key: value for key, value in metrics.items() if key != "coalition_clients"},
                "loo_value": loo_value,
            }
        )

    raw_normalized_contributions = build_normalized_contributions(loo_values)
    adaptive_beta, uniform_deviation = compute_adaptive_shrinkage_beta(
        raw_normalized_contributions,
        args.loo_shrinkage_gamma,
    )
    stabilized_contributions = apply_uniform_shrinkage(
        raw_normalized_contributions,
        adaptive_beta,
    )
    client_loo_contributions = build_client_loo_contributions(
        loo_values,
        raw_normalized_contributions,
        stabilized_contributions,
        client_configs,
    )

    print(
        "Adaptive LOO shrinkage:",
        f"uniform_deviation={uniform_deviation:.6f}",
        f"beta={adaptive_beta:.6f}",
        f"gamma={args.loo_shrinkage_gamma:.6f}",
    )

    payload = {
        "config": {
            "code_version": CODE_VERSION,
            "base_code_version": base.CODE_VERSION,
            "dataset": "CIFAR-10",
            "partition": f"CONTROLLED_{args.scenario_tag}",
            "scenario_tag": args.scenario_tag,
            "num_clients": args.num_clients,
            "clients_per_round": args.clients_per_round,
            "rounds": args.rounds,
            "task_size": len(task_indices),
            "gen_test_size": len(gen_test_indices),
            "private_train_size": len(private_indices),
            "fixed_seed": CONTROLLED_FIXED_SEED,
            "loo_scope": "global",
            "loo_utility_metric": args.loo_utility_metric,
            "normalized_contribution": "adaptive_uniform_shrinkage_over_positive_loo",
            "raw_normalized_contribution": "positive_loo_sum_to_one_else_uniform",
            "loo_shrinkage_gamma": args.loo_shrinkage_gamma,
            "run_count": run_count,
            "requested_client_samples": sum(client_data_sizes),
            "unused_private_samples": len(unused_private_indices),
            "controlled_note": (
                "Client data generation matches the controlled SmartFL scripts. A full federated training run "
                "is executed once using all 6 clients, then once for each leave-one-out client set. Global LOO "
                "values are computed as U(N) - U(N\\{i}) using the final model utility. Positive LOO values are "
                "first normalized to sum to one, then adaptively shrunk toward the uniform distribution using "
                "beta=D(q,u)^gamma to improve stability in weak-signal settings."
            ),
        },
        "adaptive_shrinkage": {
            "distribution_reference": "uniform",
            "uniform_distribution_value": (1.0 / args.num_clients) if args.num_clients > 0 else None,
            "gamma": args.loo_shrinkage_gamma,
            "uniform_deviation": uniform_deviation,
            "beta": adaptive_beta,
        },
        "full_coalition_metrics": {
            "utility_metric": args.loo_utility_metric,
            **{key: value for key, value in full_metrics.items() if key not in ("coalition_clients", "utility")},
            "utility": full_metrics["utility"],
        },
        "leave_one_out_results": leave_one_out_results,
        "client_average_contributions": client_loo_contributions,
        "category_average_contributions": build_category_average_contributions(client_loo_contributions),
        "raw_category_average_contributions": build_category_average_contributions(
            client_loo_contributions,
            contribution_key="raw_normalized_contribution",
        ),
        "client_configs": [
            {
                **config,
                "target_class_counts": {
                    str(class_id): int(count)
                    for class_id, count in sorted(client_target_class_counts[client_id].items())
                },
            }
            for client_id, config in enumerate(client_configs)
        ],
    }

    output_path = base.resolve_output_path(args.output_json)
    base.write_json_snapshot(payload, output_path)
    print(f"Controlled global LOO JSON saved to: {output_path}")


def main():
    args = parse_args()
    run_loo_experiment(args)


if __name__ == "__main__":
    main()
