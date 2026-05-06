import itertools
import math
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


CODE_VERSION = "shapley_controlled_compare_s1_global_exact_v3"
DEFAULT_OUTPUT_JSON = Path("outputs/shapley_controlled_compare/s1_global_exact_results.json")
EPS = 1e-8
PREFERRED_CATEGORY_ORDER = ["clean", "noise20", "noise40", "noise60", "iid_shared_distribution"]
UTILITY_METRIC_CHOICES = ["gen_acc", "task_acc", "neg_gen_loss", "neg_task_loss"]
CONTROLLED_FIXED_SEED = getattr(controlled_base, "CONTROLLED_FIXED_SEED", getattr(base, "FIXED_SEED", 42))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled CIFAR-10 exact Shapley baseline with configurable client size, noise rate, and missing classes"
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
        "--shapley-utility-metric",
        type=str,
        default="gen_acc",
        choices=UTILITY_METRIC_CHOICES,
    )
    parser.add_argument(
        "--include-coalition-results",
        action="store_true",
        help="Include final per-coalition metrics in the output JSON",
    )
    parser.add_argument(
        "--verbose-round-logs",
        action="store_true",
        help="Print per-round progress inside each coalition",
    )
    parser.add_argument(
        "--print-missing-classes-on-round1",
        action="store_true",
        help="Print controlled missing-class settings before training starts",
    )
    args = parser.parse_args()
    if args.clients_per_round <= 0:
        args.clients_per_round = args.num_clients
    return args


def print_shapley_setup_banner(
    args,
    private_indices,
    task_indices,
    gen_test_indices,
    device,
    requested_client_samples,
    unused_private_samples,
):
    print(
        "Controlled global-exact-Shapley setup:",
        f"code_version={CODE_VERSION}",
        "dataset=CIFAR-10",
        f"scenario={args.scenario_tag}",
        "training_aggregation=FedAvg(sample_count_weighted)",
        "contribution_method=global_exact_shapley",
        f"shapley_utility_metric={args.shapley_utility_metric}",
        "shapley_scope=whole_training_run",
        "normalized_contribution=positive_shapley_sum_to_one_else_uniform",
        f"clients={args.num_clients}",
        f"clients_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"client_train_total={requested_client_samples}",
        f"private_pool={len(private_indices)}",
        f"unused_private_pool={unused_private_samples}",
        f"eval_task_set={len(task_indices)}",
        f"eval_gen_test={len(gen_test_indices)}",
        f"coalition_count={2 ** args.num_clients}",
        f"fixed_seed={CONTROLLED_FIXED_SEED}",
        f"device={device}",
    )


def build_category_average_contributions(client_contributions):
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
            float(item["average_normalized_contribution"])
            for item in client_contributions
            if item.get("client_category", "unknown") == category
            and item.get("average_normalized_contribution") is not None
        ]
        results.append(
            {
                "client_category": category,
                "average_normalized_contribution": (sum(values) / len(values)) if values else None,
            }
        )
    return results


def build_normalized_contributions(shapley_values):
    positive_parts = [max(value, 0.0) for value in shapley_values]
    positive_sum = sum(positive_parts)
    if positive_sum > EPS:
        return [value / positive_sum for value in positive_parts]
    if not shapley_values:
        return []
    uniform_value = 1.0 / len(shapley_values)
    return [uniform_value for _ in shapley_values]


def evaluate_for_utility_metric(model, utility_metric, task_eval_loader, gen_test_loader, device):
    if utility_metric in ("gen_acc", "neg_gen_loss"):
        gen_loss, gen_acc = base.evaluate(model, gen_test_loader, device)
        return {"gen_loss": gen_loss, "gen_acc": gen_acc}
    if utility_metric in ("task_acc", "neg_task_loss"):
        task_loss, task_acc = base.evaluate(model, task_eval_loader, device)
        return {"task_loss": task_loss, "task_acc": task_acc}
    raise ValueError(f"Unsupported shapley utility metric: {utility_metric}")


def format_metric_summary(metrics, utility_metric):
    if utility_metric == "gen_acc":
        return f"gen_acc={metrics['gen_acc']:.4f}"
    if utility_metric == "task_acc":
        return f"task_acc={metrics['task_acc']:.4f}"
    if utility_metric == "neg_gen_loss":
        return f"gen_loss={metrics['gen_loss']:.4f}"
    if utility_metric == "neg_task_loss":
        return f"task_loss={metrics['task_loss']:.4f}"
    raise ValueError(f"Unsupported shapley utility metric: {utility_metric}")


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


def run_fedavg_training_for_coalition(
    coalition_clients,
    args,
    client_loaders,
    client_sizes,
    initial_global_state,
    task_eval_loader,
    gen_test_loader,
    device,
    coalition_label,
):
    coalition_clients = list(sorted(coalition_clients))
    coalition_model = base.build_model().to(device)
    coalition_model.load_state_dict(initial_global_state)

    if not coalition_clients:
        metrics = evaluate_for_utility_metric(
            coalition_model, args.shapley_utility_metric, task_eval_loader, gen_test_loader, device
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
                f"  {coalition_label} round {round_idx:03d}/{args.rounds:03d} "
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

    metrics = evaluate_for_utility_metric(
        coalition_model, args.shapley_utility_metric, task_eval_loader, gen_test_loader, device
    )
    return {"coalition_clients": coalition_clients, **metrics}


def compute_utility_from_metrics(metrics, empty_metrics, utility_metric):
    if utility_metric == "gen_acc":
        return float(metrics["gen_acc"] - empty_metrics["gen_acc"])
    if utility_metric == "task_acc":
        return float(metrics["task_acc"] - empty_metrics["task_acc"])
    if utility_metric == "neg_gen_loss":
        return float(empty_metrics["gen_loss"] - metrics["gen_loss"])
    if utility_metric == "neg_task_loss":
        return float(empty_metrics["task_loss"] - metrics["task_loss"])
    raise ValueError(f"Unsupported shapley utility metric: {utility_metric}")


def build_coalition_results(
    args,
    client_loaders,
    client_sizes,
    initial_global_state,
    task_eval_loader,
    gen_test_loader,
    device,
):
    coalition_results = {}
    coalition_payload = [] if args.include_coalition_results else None
    all_client_ids = list(range(args.num_clients))
    coalition_count = 2 ** args.num_clients
    coalition_index = 0
    empty_metrics = None

    for coalition_size in range(args.num_clients + 1):
        for coalition_clients in itertools.combinations(all_client_ids, coalition_size):
            coalition_index += 1
            coalition_label = f"[coalition {coalition_index:02d}/{coalition_count:02d}]"
            print(
                f"{coalition_label} start clients={list(coalition_clients)}"
            )
            metrics = run_fedavg_training_for_coalition(
                coalition_clients=coalition_clients,
                args=args,
                client_loaders=client_loaders,
                client_sizes=client_sizes,
                initial_global_state=initial_global_state,
                task_eval_loader=task_eval_loader,
                gen_test_loader=gen_test_loader,
                device=device,
                coalition_label=coalition_label,
            )
            if empty_metrics is None:
                empty_metrics = metrics
                metrics["utility"] = 0.0
            else:
                metrics["utility"] = compute_utility_from_metrics(
                    metrics, empty_metrics, args.shapley_utility_metric
                )
            coalition_results[tuple(coalition_clients)] = metrics
            print(
                f"{coalition_label} finished: "
                f"{format_metric_summary(metrics, args.shapley_utility_metric)} "
                f"utility={metrics['utility']:.6f}"
            )
            if coalition_payload is not None:
                coalition_payload.append(
                    {
                        "coalition_clients": list(coalition_clients),
                        "coalition_size": len(coalition_clients),
                        **{key: value for key, value in metrics.items() if key != "coalition_clients"},
                    }
                )

    return coalition_results, coalition_payload


def compute_exact_shapley_values(coalition_results, num_clients):
    factorial = [math.factorial(k) for k in range(num_clients + 1)]
    normalizer = factorial[num_clients]
    shapley_values = [0.0 for _ in range(num_clients)]

    for client_id in range(num_clients):
        other_clients = [idx for idx in range(num_clients) if idx != client_id]
        for subset_size in range(len(other_clients) + 1):
            for subset in itertools.combinations(other_clients, subset_size):
                subset_key = tuple(sorted(subset))
                superset_key = tuple(sorted(subset + (client_id,)))
                utility_without = coalition_results[subset_key]["utility"]
                utility_with = coalition_results[superset_key]["utility"]
                coefficient = (
                    factorial[subset_size]
                    * factorial[num_clients - subset_size - 1]
                    / normalizer
                )
                shapley_values[client_id] += coefficient * (utility_with - utility_without)

    return shapley_values


def build_client_shapley_contributions(shapley_values, normalized_contributions, client_metadata):
    client_contributions = []
    for client_id, shapley_value in enumerate(shapley_values):
        client_contributions.append(
            {
                "client_id": client_id,
                **client_metadata[client_id],
                "shapley_value": float(shapley_value),
                "shapley_positive_part": float(max(shapley_value, 0.0)),
                "average_normalized_contribution": float(normalized_contributions[client_id]),
            }
        )
    return client_contributions


def run_shapley_experiment(args):
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be <= num_clients")
    if args.shapley_utility_metric not in UTILITY_METRIC_CHOICES:
        raise ValueError(
            f"Unsupported shapley utility metric {args.shapley_utility_metric}. "
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

    print_shapley_setup_banner(
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

    coalition_results, coalition_payload = build_coalition_results(
        args=args,
        client_loaders=client_loaders,
        client_sizes=client_sizes,
        initial_global_state=initial_global_state,
        task_eval_loader=task_eval_loader,
        gen_test_loader=gen_test_loader,
        device=device,
    )
    shapley_values = compute_exact_shapley_values(coalition_results, args.num_clients)
    normalized_contributions = build_normalized_contributions(shapley_values)
    client_shapley_contributions = build_client_shapley_contributions(
        shapley_values,
        normalized_contributions,
        client_configs,
    )

    empty_metrics = coalition_results[()]
    grand_metrics = coalition_results[tuple(range(args.num_clients))]

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
            "shapley_scope": "global_exact",
            "shapley_utility_metric": args.shapley_utility_metric,
            "normalized_contribution": "positive_shapley_sum_to_one_else_uniform",
            "coalition_count": 2 ** args.num_clients,
            "requested_client_samples": sum(client_data_sizes),
            "unused_private_samples": len(unused_private_indices),
            "controlled_note": (
                "Client data generation matches the controlled SmartFL scripts. For every coalition of the 6 "
                "clients, the full federated training process is rerun from the same initialization, then exact "
                "Shapley values are computed from the final coalition utility."
            ),
        },
        "empty_coalition_metrics": {
            "utility_metric": args.shapley_utility_metric,
            **{key: value for key, value in empty_metrics.items() if key not in ("coalition_clients", "utility")},
            "utility": empty_metrics["utility"],
        },
        "grand_coalition_metrics": {
            "utility_metric": args.shapley_utility_metric,
            **{key: value for key, value in grand_metrics.items() if key not in ("coalition_clients", "utility")},
            "utility": grand_metrics["utility"],
        },
        "client_average_contributions": client_shapley_contributions,
        "category_average_contributions": build_category_average_contributions(client_shapley_contributions),
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
    if coalition_payload is not None:
        payload["coalition_results"] = coalition_payload

    output_path = base.resolve_output_path(args.output_json)
    base.write_json_snapshot(payload, output_path)
    print(f"Controlled global exact Shapley JSON saved to: {output_path}")


def main():
    args = parse_args()
    run_shapley_experiment(args)


if __name__ == "__main__":
    main()
