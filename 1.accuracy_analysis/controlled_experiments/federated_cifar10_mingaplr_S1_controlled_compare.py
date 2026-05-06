import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SIBLING_CIFAR10_DIR = REPO_ROOT / "1.accuracy_analysis" / "cifar10"
for extra_path in (REPO_ROOT, SIBLING_CIFAR10_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

import federated_cifar10_mingaplr_compare as base
import federated_cifar10_hist_mingaplr_base as hist_base


CODE_VERSION = "mingaplr_controlled_compare_s1_default_v1"
DEFAULT_OUTPUT_JSON = Path("outputs/mingaplr_controlled_compare/s1_results.json")


class FixedLabelSubset(Dataset):
    def __init__(self, dataset, indices, labels):
        self.dataset = dataset
        self.indices = list(indices)
        self.labels = [int(label) for label in labels]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, _ = self.dataset[self.indices[idx]]
        return image, self.labels[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled CIFAR-10 SmartFL experiment with configurable client size, noise rate, and missing classes"
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
    args = parser.parse_args()
    if args.clients_per_round <= 0:
        args.clients_per_round = args.num_clients
    return args


def parse_per_client_values(raw_value, num_clients, value_name, caster):
    tokens = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{value_name} cannot be empty")
    if len(tokens) == 1:
        return [caster(tokens[0]) for _ in range(num_clients)]
    if len(tokens) != num_clients:
        raise ValueError(
            f"{value_name} must provide either 1 value or exactly {num_clients} values, got {len(tokens)}"
        )
    return [caster(token) for token in tokens]


def cast_client_size(token):
    value = int(token)
    if value <= 0:
        raise ValueError("client data size must be positive")
    return value


def cast_noise_rate(token):
    value = float(token)
    if value > 1.0:
        value = value / 100.0
    if value < 0.0 or value > 1.0:
        raise ValueError("noise rate must be in [0, 1] or expressed as a percentage in [0, 100]")
    return value


def cast_missing_classes(token):
    value = int(token)
    if value < 0 or value >= 10:
        raise ValueError("missing class count must be in [0, 9]")
    return value


def build_missing_class_map(num_classes, missing_counts):
    preset_missing_class_map = {
        1: [0],
        2: [0, 1],
        3: [1, 2, 3],
        4: [2, 3, 4, 5],
        5: [3, 4, 5, 6, 7],
    }
    missing_class_map = {}
    next_start = 0
    for missing_count in missing_counts:
        if missing_count <= 0 or missing_count in missing_class_map:
            continue
        if missing_count in preset_missing_class_map:
            missing_classes = list(preset_missing_class_map[missing_count])
            if any(class_id < 0 or class_id >= num_classes for class_id in missing_classes):
                raise ValueError(
                    f"Preset missing-class mapping for missing_count={missing_count} exceeds available classes."
                )
            missing_class_map[missing_count] = missing_classes
            continue
        missing_class_map[missing_count] = [
            (next_start + offset) % num_classes for offset in range(missing_count)
        ]
        next_start = (next_start + missing_count + 1) % num_classes
    return missing_class_map


def choose_available_classes(num_classes, client_id, missing_count, seed, missing_class_map=None):
    del client_id
    del seed
    class_ids = list(range(num_classes))
    if missing_count <= 0:
        return class_ids, []
    if missing_class_map is not None and missing_count in missing_class_map:
        missing_classes = list(missing_class_map[missing_count])
    else:
        missing_classes = class_ids[:missing_count]
    available_classes = [class_id for class_id in class_ids if class_id not in missing_classes]
    return available_classes, missing_classes


def build_controlled_client_indices(private_indices, private_targets, client_sizes, missing_counts, seed):
    private_indices = np.asarray(private_indices, dtype=np.int64)
    private_targets = np.asarray(private_targets, dtype=np.int64)
    class_ids = sorted(np.unique(private_targets).tolist())
    num_classes = len(class_ids)
    rng = np.random.default_rng(seed)

    class_pools = {}
    for class_id in class_ids:
        class_pool = private_indices[private_targets == class_id].copy()
        rng.shuffle(class_pool)
        class_pools[class_id] = class_pool.tolist()

    missing_class_map = build_missing_class_map(num_classes, missing_counts)

    client_target_class_counts = []
    client_available_classes = []
    client_missing_class_lists = []
    total_requested_by_class = {class_id: 0 for class_id in class_ids}

    for client_id, client_size in enumerate(client_sizes):
        available_classes, missing_classes = choose_available_classes(
            num_classes, client_id, missing_counts[client_id], seed, missing_class_map
        )
        if not available_classes:
            raise ValueError(f"Client p{client_id + 1} has no available classes after applying missing-class setting.")

        base_count = client_size // len(available_classes)
        remainder = client_size % len(available_classes)
        class_counts = {}
        for offset, class_id in enumerate(available_classes):
            take = base_count + (1 if offset < remainder else 0)
            class_counts[class_id] = take
            total_requested_by_class[class_id] += take

        client_target_class_counts.append(class_counts)
        client_available_classes.append(available_classes)
        client_missing_class_lists.append(missing_classes)

    for class_id in class_ids:
        available = len(class_pools[class_id])
        requested = total_requested_by_class[class_id]
        if requested > available:
            raise ValueError(
                f"Requested {requested} samples from class {class_id}, but only {available} are available "
                "after task/gen_test splitting. Reduce client size(s) or missing-class pressure."
            )

    client_indices = []
    for client_id, class_counts in enumerate(client_target_class_counts):
        allocated = []
        for class_id in sorted(class_counts.keys()):
            take = class_counts[class_id]
            allocated.extend(class_pools[class_id][:take])
            del class_pools[class_id][:take]
        rng.shuffle(allocated)
        client_indices.append(allocated)

    unused_private_indices = []
    for class_id in class_ids:
        unused_private_indices.extend(class_pools[class_id])
    rng.shuffle(unused_private_indices)

    return client_indices, client_available_classes, client_missing_class_lists, client_target_class_counts, unused_private_indices


def build_fixed_noisy_labels(original_labels, noise_rate, seed):
    labels = np.asarray(original_labels, dtype=np.int64).copy()
    if noise_rate <= 0.0 or labels.size == 0:
        return labels.tolist()

    unique_classes = sorted(np.unique(labels).tolist())
    if len(unique_classes) <= 1:
        return labels.tolist()

    rng = np.random.default_rng(seed)
    flip_mask = rng.random(labels.shape[0]) < noise_rate
    if not np.any(flip_mask):
        return labels.tolist()

    replacement_map = {}
    for class_id in unique_classes:
        replacement_map[class_id] = [candidate for candidate in unique_classes if candidate != class_id]

    for sample_idx in np.where(flip_mask)[0]:
        current_label = int(labels[sample_idx])
        labels[sample_idx] = int(rng.choice(replacement_map[current_label]))

    return labels.tolist()


def build_controlled_client_loaders(dataset, full_targets, client_indices, client_noise_rates, batch_size, loader_kwargs, base_seed):
    client_loaders = []
    client_sizes = []
    full_targets = np.asarray(full_targets, dtype=np.int64)

    for client_id, indices in enumerate(client_indices):
        noisy_labels = build_fixed_noisy_labels(
            full_targets[np.asarray(indices, dtype=np.int64)],
            client_noise_rates[client_id],
            base_seed + client_id,
        )
        subset = FixedLabelSubset(dataset, indices, noisy_labels)
        generator = torch.Generator()
        generator.manual_seed(base_seed + client_id)
        client_loader_kwargs = dict(loader_kwargs)
        client_loader_kwargs["generator"] = generator
        client_loaders.append(base.DataLoader(subset, batch_size=batch_size, **client_loader_kwargs))
        client_sizes.append(len(indices))

    return client_loaders, client_sizes


def build_client_configs(client_sizes, client_noise_rates, client_available_classes, client_missing_classes, client_missing_class_lists):
    client_configs = []
    for client_id in range(len(client_sizes)):
        noise_rate = float(client_noise_rates[client_id])
        missing_class_count = int(client_missing_classes[client_id])
        if noise_rate > 0.0:
            category = f"noise{int(round(noise_rate * 100))}"
        elif missing_class_count > 0:
            category = f"missing{missing_class_count}"
        else:
            category = "iid_shared_distribution"
        client_configs.append(
            {
                "client_id": client_id,
                "participant_id": f"p{client_id + 1}",
                "client_size": int(client_sizes[client_id]),
                "noise_rate": noise_rate,
                "missing_class_count": missing_class_count,
                "client_category": category,
                "available_classes": [int(class_id) for class_id in client_available_classes[client_id]],
                "missing_classes": [int(class_id) for class_id in client_missing_class_lists[client_id]],
            }
        )
    return client_configs


def format_client_configs_for_banner(client_configs):
    tokens = []
    for config in client_configs:
        tokens.append(
            f"{config['participant_id']}:size={config['client_size']},"
            f"noise={int(round(config['noise_rate'] * 100))}%,"
            f"missing={config['missing_class_count']}"
        )
    return " | ".join(tokens)


def build_controlled_selected_clients_info(selected_clients, client_metadata):
    tokens = []
    for client_id in selected_clients:
        metadata = client_metadata[client_id]
        tokens.append(
            f"p{client_id + 1}:noise={int(round(100 * metadata['noise_rate']))}%"
            f"/missing={metadata['missing_class_count']}"
        )
    return f"controlled_clients={tokens}" if tokens else ""


def build_selected_clients_missing_classes_info(selected_clients, client_metadata):
    tokens = []
    for client_id in selected_clients:
        missing_classes = client_metadata[client_id].get("missing_classes", [])
        rendered = ",".join(str(class_id) for class_id in missing_classes) if missing_classes else "none"
        tokens.append(f"p{client_id + 1}:[{rendered}]")
    return " | ".join(tokens)


def run_federated_learning(args):
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be <= num_clients")

    client_data_sizes = parse_per_client_values(
        args.client_data_sizes, args.num_clients, "client-data-sizes", cast_client_size
    )
    client_noise_rates = parse_per_client_values(
        args.client_noise_rates, args.num_clients, "client-noise-rates", cast_noise_rate
    )
    client_missing_classes = parse_per_client_values(
        args.client_missing_classes, args.num_clients, "client-missing-classes", cast_missing_classes
    )

    base.set_seed(base.FIXED_SEED)
    devices = base.resolve_devices(args)
    device = devices[0]

    train_dataset, test_dataset = base.load_cifar10(args.data_dir)
    full_dataset = base.ConcatDataset([train_dataset, test_dataset])
    full_targets = base.np.asarray(train_dataset.targets + test_dataset.targets, dtype=base.np.int64)
    private_indices, task_indices, gen_test_indices = base.stratified_three_way_split_indices(
        full_targets,
        args.task_size,
        args.gen_test_size,
        base.FIXED_SEED,
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
    ) = build_controlled_client_indices(
        private_indices,
        private_targets,
        client_data_sizes,
        client_missing_classes,
        base.FIXED_SEED,
    )
    client_configs = build_client_configs(
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

    client_loaders, client_sizes = build_controlled_client_loaders(
        full_dataset,
        full_targets,
        client_indices,
        client_noise_rates,
        args.batch_size,
        client_loader_kwargs,
        base.FIXED_SEED,
    )

    task_subset = base.Subset(full_dataset, task_indices.tolist())
    server_proxy_batch_size = len(task_subset)
    server_proxy_batches_per_round = 1
    task_loader = base.DataLoader(task_subset, batch_size=server_proxy_batch_size, **train_loader_kwargs)
    task_iterator = iter(task_loader)
    task_eval_loader = base.DataLoader(task_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    gen_test_subset = base.Subset(full_dataset, gen_test_indices.tolist())
    gen_test_loader = base.DataLoader(gen_test_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    hist_base.print_setup_banner(
        code_version=CODE_VERSION,
        partition_name=f"CONTROLLED_{args.scenario_tag}",
        args=args,
        private_indices=private_indices,
        task_indices=task_indices,
        gen_test_indices=gen_test_indices,
        device=device,
        startup_tokens=[
            f"scenario={args.scenario_tag}",
            "client_partition=controlled",
            "class_distribution=equal_over_available_classes_per_client",
            "noise_flip_scope=within_client_existing_classes_only",
            f"requested_client_samples={sum(client_data_sizes)}",
            f"unused_private_samples={len(unused_private_indices)}",
            f"server_proxy_batch_size={server_proxy_batch_size}",
            f"server_proxy_batches_per_round={server_proxy_batches_per_round}",
        ],
    )
    print("Controlled clients:", format_client_configs_for_banner(client_configs))
    if getattr(args, "print_missing_classes_on_round1", False):
        print(
            "Controlled missing classes:",
            build_selected_clients_missing_classes_info(range(args.num_clients), client_configs),
        )

    global_model = base.build_model().to(device)
    round_records = []

    for round_idx in range(1, args.rounds + 1):
        selected_clients = base.random.sample(range(args.num_clients), args.clients_per_round)
        selected_info = build_controlled_selected_clients_info(selected_clients, client_configs)
        print(f"Round {round_idx:03d} selected_clients={selected_clients} {selected_info}")
        task_batches, task_iterator = base.next_batches(task_loader, task_iterator, server_proxy_batches_per_round)
        global_snapshot = base.clone_state_dict_to_cpu(global_model.state_dict())

        trial_model = base.build_model().to(device)
        trial_model.load_state_dict(global_snapshot)
        round_result = hist_base.run_round_on_device_with_analysis(
            round_idx,
            device,
            args,
            selected_clients,
            client_loaders,
            client_sizes,
            client_configs,
            trial_model,
            task_batches,
            task_eval_loader,
            gen_test_loader,
        )
        global_model = round_result["global_model"]
        round_records.append(round_result["round_record"])

        print(
            f"Round {round_idx:03d}:",
            f"selected_clients={selected_clients}",
            f"proxy_loss={round_result['round_record']['proxy_loss_post_optimization']:.4f}",
            f"server_steps_used={round_result['round_record']['server_steps_used']}",
            f"server_early_stopped={round_result['round_record']['server_early_stopped']}",
            f"server_stop_reason={round_result['round_record']['server_stop_reason']}",
            f"task_loss={round_result['round_record']['task_loss_post_optimization']:.4f}",
            f"task_acc={round_result['round_record']['task_acc_post_optimization']:.4f}",
            f"gen_loss={round_result['round_record']['gen_loss_post_optimization']:.4f}",
            f"gen_acc={round_result['round_record']['gen_acc_post_optimization']:.4f}",
        )

    output_path = base.resolve_output_path(args.output_json)
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
            "fixed_seed": base.FIXED_SEED,
            "window_size": hist_base.WINDOW_SIZE,
            "grad_damping": args.grad_damping,
            "server_lr_cap": args.server_lr_cap,
            "server_opt_steps": args.server_opt_steps,
            "requested_client_samples": sum(client_data_sizes),
            "unused_private_samples": len(unused_private_indices),
            "client_size_unique_values": sorted(set(client_sizes)),
            "client_size_histogram": {
                str(size): sum(1 for client_size in client_sizes if client_size == size)
                for size in sorted(set(client_sizes))
            },
            "controlled_note": (
                "When missing_class_count=0 and all client sizes are equal, each client receives an equal number "
                "of samples from every class, yielding absolute IID class proportions. Noise is fixed at "
                "initialization and label flips stay within each client's available classes."
            ),
        },
        "rounds": round_records,
        "round_gradient_metrics": hist_base.build_round_gradient_metrics(round_records),
        "window_average_weights": hist_base.build_window_average_weights(round_records, hist_base.WINDOW_SIZE),
        "client_average_weights": hist_base.build_client_average_weights(
            round_records,
            args.num_clients,
            args.rounds,
            client_configs,
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
    base.write_json_snapshot(payload, output_path)
    print(f"Controlled hist-compatible JSON saved to: {output_path}")


def main():
    args = parse_args()
    run_federated_learning(args)


if __name__ == "__main__":
    main()
