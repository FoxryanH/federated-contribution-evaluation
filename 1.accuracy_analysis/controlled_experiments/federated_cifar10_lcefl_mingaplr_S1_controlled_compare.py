import argparse
import math
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
import federated_cifar10_mingaplr_S1_controlled_compare as controlled_base


CODE_VERSION = "lcefl_controlled_compare_s1_v1"
DEFAULT_OUTPUT_JSON = Path("outputs/lcefl_controlled_compare/s1_results5.json")
DEFAULT_LCEFL_RE = 0.4
DEFAULT_LCEFL_NM = 1
EPS = 1e-12
CONTROLLED_FIXED_SEED = getattr(controlled_base, "CONTROLLED_FIXED_SEED", getattr(base, "FIXED_SEED", 42))
WINDOW_SIZE = 10
PREFERRED_CATEGORY_ORDER = ["clean", "noise20", "noise40", "noise60", "iid_shared_distribution"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled CIFAR-10 LCEFL baseline with configurable client size, noise rate, and missing classes"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--scenario-tag", type=str, default="S1")
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=-1,
        help="Set to -1 to select all clients every round",
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
        "--lcefl-re",
        type=float,
        default=DEFAULT_LCEFL_RE,
        help="Representative layer selection ratio r_e",
    )
    parser.add_argument(
        "--lcefl-nm",
        type=int,
        default=DEFAULT_LCEFL_NM,
        help="Number of parameter tensors within each group n_m",
    )
    args = parser.parse_args()
    if args.clients_per_round <= 0:
        args.clients_per_round = args.num_clients
    if args.lcefl_re <= 0.0 or args.lcefl_re > 1.0:
        raise ValueError("lcefl-re must be in (0, 1].")
    if args.lcefl_nm <= 0:
        raise ValueError("lcefl-nm must be a positive integer.")
    return args


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


def normalize_contributions(values):
    if not values:
        return []
    total = float(sum(values))
    if math.isfinite(total) and abs(total) > EPS:
        return [float(value / total) for value in values]
    uniform_value = 1.0 / len(values)
    return [uniform_value for _ in values]


def build_window_average_contributions(round_raw_contributions, window_size, client_metadata):
    if not round_raw_contributions:
        return []

    encountered_categories = []
    for metadata in client_metadata:
        category = metadata.get("client_category", "unknown")
        if category not in encountered_categories:
            encountered_categories.append(category)
    category_order = [category for category in PREFERRED_CATEGORY_ORDER if category in encountered_categories]
    category_order.extend(
        [category for category in encountered_categories if category not in PREFERRED_CATEGORY_ORDER]
    )

    windows = []
    num_clients = len(client_metadata)
    for start in range(0, len(round_raw_contributions), window_size):
        window_records = round_raw_contributions[start : start + window_size]
        window_totals = [0.0 for _ in range(num_clients)]
        for round_values in window_records:
            for client_id in range(num_clients):
                window_totals[client_id] += float(round_values[client_id])
        normalized_window = normalize_contributions(window_totals)
        category_means = {}
        for category in category_order:
            category_values = [
                normalized_window[client_id]
                for client_id, metadata in enumerate(client_metadata)
                if metadata.get("client_category", "unknown") == category
            ]
            if category_values:
                category_means[category] = float(sum(category_values) / len(category_values))
        windows.append(
            {
                "window_start_round": start + 1,
                "window_end_round": start + len(window_records),
                "average_normalized_contribution": category_means,
            }
        )
    return windows


def get_parameter_layer_names(model):
    return [name for name, _ in model.named_parameters()]


def compute_layer_sensitivity(global_prev_state, global_curr_state, layer_name):
    delta = global_curr_state[layer_name].to(torch.float64) - global_prev_state[layer_name].to(torch.float64)
    return float(torch.linalg.vector_norm(delta).item())


def select_lcefl_layers(global_prev_state, global_curr_state, parameter_layer_names, members_per_group, selection_ratio):
    groups = []
    for start in range(0, len(parameter_layer_names), members_per_group):
        group_layers = parameter_layer_names[start : start + members_per_group]
        if group_layers:
            groups.append(group_layers)

    representative_layers = []
    for group_layers in groups:
        representative = group_layers[0]
        representative_layers.append(
            {
                "layer_name": representative,
                "group_layers": list(group_layers),
                "sensitivity": compute_layer_sensitivity(global_prev_state, global_curr_state, representative),
            }
        )

    representative_layers.sort(key=lambda item: item["sensitivity"], reverse=True)
    selected_count = max(1, int(math.ceil(len(representative_layers) * selection_ratio)))
    selected_layers = representative_layers[:selected_count]
    return selected_layers, representative_layers


def build_selected_layer_vector(target_state, reference_state, selected_layer_names):
    parts = []
    for layer_name in selected_layer_names:
        delta = target_state[layer_name].to(torch.float64) - reference_state[layer_name].to(torch.float64)
        parts.append(delta.reshape(-1))
    if not parts:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(parts, dim=0)


def execute_fedavg_round(
    round_idx,
    selected_clients,
    device,
    args,
    client_loaders,
    client_sizes,
    global_model,
    task_eval_loader,
    gen_test_loader,
):
    global_model = global_model.to(device)
    global_state_before = base.clone_state_dict_to_cpu(global_model.state_dict())

    local_states = []
    sample_counts = []
    for client_id in selected_clients:
        local_seed = CONTROLLED_FIXED_SEED + round_idx * 1000 + client_id
        local_state, sample_count = base.client_local_train(
            global_state_before,
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
    next_state = base.aggregate_states_on_cpu(local_states, aggregation_weights, global_model)
    global_model.load_state_dict(next_state)
    global_state_after = base.clone_state_dict_to_cpu(global_model.state_dict())

    task_loss, task_acc = base.evaluate(global_model, task_eval_loader, device)
    gen_loss, gen_acc = base.evaluate(global_model, gen_test_loader, device)

    return {
        "global_model": global_model,
        "global_state_before": global_state_before,
        "global_state_after": global_state_after,
        "local_states": local_states,
        "sample_counts": sample_counts,
        "task_loss_post_optimization": task_loss,
        "task_acc_post_optimization": task_acc,
        "gen_loss_post_optimization": gen_loss,
        "gen_acc_post_optimization": gen_acc,
        "aggregation_weights": aggregation_weights,
    }


def build_pass_loaders(
    args,
    full_dataset,
    full_targets,
    client_indices,
    client_noise_rates,
    task_indices,
    gen_test_indices,
):
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

    return {
        "client_loaders": client_loaders,
        "client_sizes": client_sizes,
        "task_eval_loader": task_eval_loader,
        "gen_test_loader": gen_test_loader,
    }


def run_training_pass_collect_metadata(
    args,
    client_configs,
    loaders,
    parameter_layer_names,
    device,
):
    global_model = base.build_model().to(device)

    round_metadata = []
    selected_clients_schedule = []

    for round_idx in range(1, args.rounds + 1):
        selected_clients = base.random.sample(range(args.num_clients), args.clients_per_round)
        selected_clients_schedule.append(list(selected_clients))
        selected_info = controlled_base.build_controlled_selected_clients_info(selected_clients, client_configs)
        print(f"[train pass 1] round {round_idx:03d} selected_clients={selected_clients} {selected_info}")

        round_result = execute_fedavg_round(
            round_idx=round_idx,
            selected_clients=selected_clients,
            device=device,
            args=args,
            client_loaders=loaders["client_loaders"],
            client_sizes=loaders["client_sizes"],
            global_model=global_model,
            task_eval_loader=loaders["task_eval_loader"],
            gen_test_loader=loaders["gen_test_loader"],
        )
        global_model = round_result["global_model"]

        selected_layers, representative_layers = select_lcefl_layers(
            round_result["global_state_before"],
            round_result["global_state_after"],
            parameter_layer_names,
            args.lcefl_nm,
            args.lcefl_re,
        )
        selected_layer_names = [item["layer_name"] for item in selected_layers]
        representative_layer_payload = [
            {
                "layer_name": item["layer_name"],
                "sensitivity": float(item["sensitivity"]),
            }
            for item in representative_layers
        ]

        round_metadata.append(
            {
                "round_index": round_idx,
                "selected_clients": list(selected_clients),
                "selected_layer_names": selected_layer_names,
                "selected_layer_count": len(selected_layer_names),
                "selected_layer_sensitivities": [
                    {
                        "layer_name": item["layer_name"],
                        "sensitivity": float(item["sensitivity"]),
                    }
                    for item in selected_layers
                ],
                "representative_layer_sensitivities": representative_layer_payload,
                "aggregation_weights": [float(weight) for weight in round_result["aggregation_weights"]],
                "task_loss_post_optimization": round_result["task_loss_post_optimization"],
                "task_acc_post_optimization": round_result["task_acc_post_optimization"],
                "gen_loss_post_optimization": round_result["gen_loss_post_optimization"],
                "gen_acc_post_optimization": round_result["gen_acc_post_optimization"],
            }
        )

        print(
            f"[train pass 1] round {round_idx:03d} done: "
            f"selected_layers={len(selected_layer_names)} "
            f"task_acc={round_result['task_acc_post_optimization']:.4f} "
            f"gen_acc={round_result['gen_acc_post_optimization']:.4f}"
        )

    final_global_state = base.clone_state_dict_to_cpu(global_model.state_dict())
    return final_global_state, round_metadata, selected_clients_schedule


def run_training_pass_compute_lcefl(
    args,
    client_configs,
    loaders,
    round_metadata,
    final_global_state,
    device,
):
    global_model = base.build_model().to(device)

    total_contributions = [0.0 for _ in range(args.num_clients)]
    selected_counts = [0 for _ in range(args.num_clients)]
    round_raw_contributions = []
    round_records = []

    for metadata in round_metadata:
        round_idx = metadata["round_index"]
        selected_clients = metadata["selected_clients"]
        selected_layer_names = metadata["selected_layer_names"]
        selected_info = controlled_base.build_controlled_selected_clients_info(selected_clients, client_configs)
        print(f"[train pass 2] round {round_idx:03d} selected_clients={selected_clients} {selected_info}")

        round_result = execute_fedavg_round(
            round_idx=round_idx,
            selected_clients=selected_clients,
            device=device,
            args=args,
            client_loaders=loaders["client_loaders"],
            client_sizes=loaders["client_sizes"],
            global_model=global_model,
            task_eval_loader=loaders["task_eval_loader"],
            gen_test_loader=loaders["gen_test_loader"],
        )
        global_model = round_result["global_model"]

        global_direction = build_selected_layer_vector(
            final_global_state,
            round_result["global_state_before"],
            selected_layer_names,
        )
        global_direction_norm = float(torch.linalg.vector_norm(global_direction).item()) if global_direction.numel() else 0.0

        round_contributions = [0.0 for _ in range(args.num_clients)]
        round_client_details = {}
        for local_idx, client_id in enumerate(selected_clients):
            local_update = build_selected_layer_vector(
                round_result["local_states"][local_idx],
                round_result["global_state_before"],
                selected_layer_names,
            )
            local_update_norm = float(torch.linalg.vector_norm(local_update).item()) if local_update.numel() else 0.0

            if local_update.numel() == 0 or global_direction.numel() == 0 or local_update_norm < EPS or global_direction_norm < EPS:
                cosine_similarity = 0.0
                projection = 0.0
                contribution = 0.0
            else:
                dot_product = float(torch.dot(local_update, global_direction).item())
                cosine_similarity = dot_product / max(local_update_norm * global_direction_norm, EPS)
                projection = dot_product / max(global_direction_norm, EPS)
                contribution = abs(cosine_similarity) * projection

            round_contributions[client_id] = float(contribution)
            total_contributions[client_id] += float(contribution)
            selected_counts[client_id] += 1
            round_client_details[client_id] = {
                "cosine_similarity": float(cosine_similarity),
                "projection": float(projection),
                "local_update_norm": float(local_update_norm),
                "global_direction_norm": float(global_direction_norm),
                "round_contribution_raw": float(contribution),
                "round_contribution": float(contribution),
            }

        round_normalized = normalize_contributions(round_contributions)
        round_raw_contributions.append(list(round_contributions))

        client_records = []
        for client_id in range(args.num_clients):
            details = round_client_details.get(
                client_id,
                {
                    "cosine_similarity": None,
                    "projection": None,
                    "local_update_norm": None,
                    "global_direction_norm": None,
                    "round_contribution_raw": 0.0,
                    "round_contribution": 0.0,
                },
            )
            client_records.append(
                {
                    "client_id": client_id,
                    **client_configs[client_id],
                    "selected_in_round": client_id in selected_clients,
                    "round_contribution_raw": float(details["round_contribution_raw"]),
                    "round_contribution": float(details["round_contribution"]),
                    "round_normalized_contribution": float(round_normalized[client_id]),
                    "cosine_similarity": details["cosine_similarity"],
                    "projection": details["projection"],
                    "local_update_norm": details["local_update_norm"],
                    "global_direction_norm": details["global_direction_norm"],
                }
            )

        round_records.append(
            {
                "round_index": round_idx,
                "selected_clients": list(selected_clients),
                "aggregation_weights": metadata["aggregation_weights"],
                "task_loss_post_optimization": metadata["task_loss_post_optimization"],
                "task_acc_post_optimization": metadata["task_acc_post_optimization"],
                "gen_loss_post_optimization": metadata["gen_loss_post_optimization"],
                "gen_acc_post_optimization": metadata["gen_acc_post_optimization"],
                "selected_layer_names": list(selected_layer_names),
                "selected_layer_count": metadata["selected_layer_count"],
                "selected_layer_sensitivities": metadata["selected_layer_sensitivities"],
                "representative_layer_sensitivities": metadata["representative_layer_sensitivities"],
                "clients": client_records,
            }
        )

        print(
            f"[train pass 2] round {round_idx:03d} done: "
            f"selected_layers={metadata['selected_layer_count']} "
            f"global_direction_norm={global_direction_norm:.6f}"
        )

    normalized_total_contributions = normalize_contributions(total_contributions)
    return round_records, round_raw_contributions, total_contributions, normalized_total_contributions, selected_counts


def print_lcefl_setup_banner(
    args,
    private_indices,
    task_indices,
    gen_test_indices,
    device,
    client_data_sizes,
    requested_client_samples,
    unused_private_samples,
):
    unique_client_sizes = sorted(set(int(size) for size in client_data_sizes))
    if len(unique_client_sizes) == 1:
        client_size_token = f"per_client_train={unique_client_sizes[0]}"
    else:
        client_size_token = f"per_client_train_set={unique_client_sizes}"

    print(
        "Controlled LCEFL setup:",
        f"code_version={CODE_VERSION}",
        "dataset=CIFAR-10",
        f"scenario={args.scenario_tag}",
        "contribution_method=LCEFL",
        "training_aggregation=FedAvg(sample_count_weighted)",
        "evaluation_scope=whole_training_trajectory_post_hoc",
        "layer_definition=parameter_tensor",
        "representative_layer=first_tensor_in_each_group",
        "layer_sensitivity_norm=l2",
        f"lcefl_re={args.lcefl_re}",
        f"lcefl_nm={args.lcefl_nm}",
        f"clients={args.num_clients}",
        f"clients_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"client_train_total={requested_client_samples}",
        client_size_token,
        f"private_pool={len(private_indices)}",
        f"unused_private_pool={unused_private_samples}",
        f"eval_task_set={len(task_indices)}",
        f"eval_gen_test={len(gen_test_indices)}",
        f"fixed_seed={CONTROLLED_FIXED_SEED}",
        f"device={device}",
    )


def run_lcefl_experiment(args):
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be <= num_clients")

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

    print_lcefl_setup_banner(
        args=args,
        private_indices=private_indices,
        task_indices=task_indices,
        gen_test_indices=gen_test_indices,
        device=device,
        client_data_sizes=client_data_sizes,
        requested_client_samples=sum(client_data_sizes),
        unused_private_samples=len(unused_private_indices),
    )
    print("Controlled clients:", controlled_base.format_client_configs_for_banner(client_configs))
    if getattr(args, "print_missing_classes_on_round1", False):
        print(
            "Controlled missing classes:",
            get_missing_classes_info(range(args.num_clients), client_configs),
        )

    parameter_layer_names = get_parameter_layer_names(base.build_model())

    base.set_seed(CONTROLLED_FIXED_SEED)
    pass1_loaders = build_pass_loaders(
        args,
        full_dataset,
        full_targets,
        client_indices,
        client_noise_rates,
        task_indices,
        gen_test_indices,
    )
    final_global_state, round_metadata, selected_clients_schedule = run_training_pass_collect_metadata(
        args=args,
        client_configs=client_configs,
        loaders=pass1_loaders,
        parameter_layer_names=parameter_layer_names,
        device=device,
    )

    base.set_seed(CONTROLLED_FIXED_SEED)
    pass2_loaders = build_pass_loaders(
        args,
        full_dataset,
        full_targets,
        client_indices,
        client_noise_rates,
        task_indices,
        gen_test_indices,
    )
    round_records, round_raw_contributions, total_contributions, normalized_total_contributions, selected_counts = (
        run_training_pass_compute_lcefl(
            args=args,
            client_configs=client_configs,
            loaders=pass2_loaders,
            round_metadata=round_metadata,
            final_global_state=final_global_state,
            device=device,
        )
    )

    client_average_contributions = []
    for client_id in range(args.num_clients):
        client_average_contributions.append(
            {
                "client_id": client_id,
                **client_configs[client_id],
                "selected_count": int(selected_counts[client_id]),
                "total_contribution": float(total_contributions[client_id]),
                "average_normalized_contribution": float(normalized_total_contributions[client_id]),
            }
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
            "window_size": WINDOW_SIZE,
            "lcefl_re": args.lcefl_re,
            "lcefl_nm": args.lcefl_nm,
            "training_aggregation": "FedAvg(sample_count_weighted)",
            "evaluation_scope": "whole_training_trajectory_post_hoc",
            "replay_passes": 2,
            "layer_definition": "parameter_tensor",
            "representative_layer_rule": "first_tensor_in_each_group",
            "layer_sensitivity_norm": "l2",
            "final_normalization": "signed_total_contribution_divided_by_signed_sum",
            "requested_client_samples": sum(client_data_sizes),
            "unused_private_samples": len(unused_private_indices),
            "client_size_unique_values": sorted(set(client_data_sizes)),
            "client_size_histogram": {
                str(size): sum(1 for client_size in client_data_sizes if client_size == size)
                for size in sorted(set(client_data_sizes))
            },
            "controlled_note": (
                "Training uses standard sample-count-weighted FedAvg without server-side weight optimization. "
                "After the full training process ends, LCEFL contributions are computed by replaying the same "
                "round schedule and projecting each round's local update onto the final global convergence "
                "direction using representative layers selected from grouped parameter tensors."
            ),
        },
        "rounds": round_records,
        "window_average_contributions": build_window_average_contributions(
            round_raw_contributions,
            WINDOW_SIZE,
            client_configs,
        ),
        "client_average_contributions": client_average_contributions,
        "category_average_contributions": build_category_average_contributions(client_average_contributions),
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
        "selected_clients_schedule": selected_clients_schedule,
    }

    output_path = base.resolve_output_path(args.output_json)
    base.write_json_snapshot(payload, output_path)
    print(f"Controlled LCEFL JSON saved to: {output_path}")


def main():
    args = parse_args()
    run_lcefl_experiment(args)


if __name__ == "__main__":
    main()
