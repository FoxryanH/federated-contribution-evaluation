import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SIBLING_CIFAR10_DIR = REPO_ROOT / "1.accuracy_analysis" / "cifar10"
for extra_path in (REPO_ROOT, SIBLING_CIFAR10_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

import federated_yahooanswers_textcnn as text_base
import federated_cifar10_mingaplr_compare as smartfl_base


FIXED_SEED = 42
CODE_VERSION = "yahooanswers_mingaplr_compare_v1"
DEFAULT_GRAD_DAMPING = smartfl_base.DEFAULT_GRAD_DAMPING
DEFAULT_CLIENT_AVG_WEIGHT_JSON = Path("outputs/yahooanswers_mingaplr_compare/client_average_weights.json")
YAHOO_ANSWERS_LABELS = text_base.YAHOO_ANSWERS_LABELS


def build_model(args):
    return text_base.build_model(args)


def set_seed(seed):
    smartfl_base.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Yahoo Answers Topics SmartFL baseline with TextCNN and min-gap-driven server learning rate"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "yahoo_answers_topics")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_CLIENT_AVG_WEIGHT_JSON)
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument(
        "--private-train-size",
        type=int,
        default=500000,
        help="Balanced private training pool size sampled from the full Yahoo Answers dataset",
    )
    parser.add_argument("--proxy-batch-size", type=int, default=1000)
    parser.add_argument(
        "--task-size",
        type=int,
        default=1000,
        help="Balanced proxy/task subset size sampled from the full Yahoo Answers dataset",
    )
    parser.add_argument(
        "--gen-test-size",
        type=int,
        default=100000,
        help="Balanced generalization test subset size sampled from the full Yahoo Answers dataset",
    )
    parser.add_argument("--proxy-batches-per-round", type=int, default=1)
    parser.add_argument("--local-optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--local-lr", type=float, default=1e-3)
    parser.add_argument("--local-momentum", type=float, default=0.0)
    parser.add_argument("--local-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--server-lr-cap",
        type=float,
        default=5.0,
        help="Upper bound for effective server learning rate; <=0 disables the cap",
    )
    parser.add_argument("--server-opt-steps", type=int, default=10)
    parser.add_argument("--server-opt-weight-tol", type=float, default=1e-4)
    parser.add_argument(
        "--grad-damping",
        type=float,
        default=DEFAULT_GRAD_DAMPING,
        help="Additive damping term used in gradient normalization denominator",
    )
    parser.add_argument(
        "--grad-stability-mean-threshold",
        type=float,
        default=0.6,
        help="Early-stop threshold for mean absolute gradient in Yahoo Answers server optimization",
    )
    parser.add_argument(
        "--grad-stability-range-threshold",
        type=float,
        default=0.4,
        help="Early-stop threshold for max-min gradient range in Yahoo Answers server optimization",
    )
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-channels", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def load_yahooanswers(args):
    train_csv = args.train_csv or (args.data_dir / text_base.TRAIN_FILE_NAME)
    test_csv = args.test_csv or (args.data_dir / text_base.TEST_FILE_NAME)
    text_base.maybe_download_yahooanswers(args.data_dir, train_csv, test_csv, args.download)

    train_samples = text_base.load_yahooanswers_csv(train_csv, args.max_train_samples)
    test_samples = text_base.load_yahooanswers_csv(test_csv, args.max_test_samples)
    if not train_samples or not test_samples:
        raise RuntimeError("Failed to load Yahoo Answers Topics samples from the provided csv files.")

    all_samples = train_samples + test_samples
    targets = np.asarray([label for _, label in all_samples], dtype=np.int64)
    dataset = text_base.YahooAnswersTopicsDataset(all_samples, args.vocab_size, args.max_length)
    return dataset, targets


def _balanced_class_quotas(total_size, class_ids):
    num_classes = len(class_ids)
    base_quota = total_size // num_classes
    remainder = total_size % num_classes
    return {
        int(class_id): base_quota + (1 if offset < remainder else 0)
        for offset, class_id in enumerate(class_ids)
    }


def balanced_three_way_split_indices(targets, private_train_size, task_size, gen_test_size, seed):
    targets = np.asarray(targets, dtype=np.int64)
    class_ids = sorted(np.unique(targets).tolist())
    rng = np.random.default_rng(seed)

    requested_sizes = {
        "private_train": int(private_train_size),
        "task": int(task_size),
        "gen_test": int(gen_test_size),
    }
    if any(size <= 0 for size in requested_sizes.values()):
        raise ValueError(
            "private_train_size, task_size, and gen_test_size must all be positive "
            f"for balanced Yahoo Answers sampling: {requested_sizes}"
        )

    quotas = {
        split_name: _balanced_class_quotas(split_size, class_ids)
        for split_name, split_size in requested_sizes.items()
    }

    per_class_indices = {}
    for class_id in class_ids:
        class_positions = np.where(targets == class_id)[0]
        rng.shuffle(class_positions)
        per_class_indices[int(class_id)] = class_positions.tolist()

    split_indices = {split_name: [] for split_name in requested_sizes}
    split_class_counts = {split_name: {} for split_name in requested_sizes}

    for class_id in class_ids:
        class_pool = per_class_indices[int(class_id)]
        required_for_class = sum(quotas[split_name][int(class_id)] for split_name in requested_sizes)
        if required_for_class > len(class_pool):
            raise ValueError(
                f"Not enough Yahoo Answers samples for class {class_id}. "
                f"Need {required_for_class}, but only found {len(class_pool)}."
            )

        cursor = 0
        for split_name in ["private_train", "task", "gen_test"]:
            take = quotas[split_name][int(class_id)]
            chosen = class_pool[cursor:cursor + take]
            split_indices[split_name].extend(chosen)
            split_class_counts[split_name][str(class_id)] = len(chosen)
            cursor += take

    for split_name in split_indices:
        rng.shuffle(split_indices[split_name])

    metadata = {
        "split_strategy": "balanced_class_sampling",
        "requested_sizes": requested_sizes,
        "actual_sizes": {split_name: len(indices) for split_name, indices in split_indices.items()},
        "class_counts": split_class_counts,
    }
    return (
        np.asarray(split_indices["private_train"], dtype=np.int64),
        np.asarray(split_indices["task"], dtype=np.int64),
        np.asarray(split_indices["gen_test"], dtype=np.int64),
        metadata,
    )


def client_local_train(global_state, dataset_loader, sample_count, device, args, local_seed):
    local_model = build_model(args).to(device)
    local_model.load_state_dict(global_state)
    local_model.train()

    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)

    optimizer = smartfl_base.build_optimizer(local_model, args)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(args.local_epochs):
        for inputs, targets in dataset_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    local_state = smartfl_base.clone_state_dict_to_cpu(local_model.state_dict())
    del optimizer
    del criterion
    del local_model
    smartfl_base.clear_device_cache(device)
    return local_state, sample_count


def run_round_on_device(
    round_idx,
    device,
    args,
    selected_clients,
    client_loaders,
    client_sizes,
    global_model,
    task_batches,
    task_eval_loader,
    gen_test_loader,
):
    global_model = global_model.to(device)
    global_state = smartfl_base.clone_state_dict_to_cpu(global_model.state_dict())

    local_states = []
    sample_counts = []
    for client_id in selected_clients:
        local_seed = FIXED_SEED + round_idx * 1000 + client_id
        local_state, sample_count = client_local_train(
            global_state,
            client_loaders[client_id],
            client_sizes[client_id],
            device,
            args,
            local_seed,
        )
        local_states.append(local_state)
        sample_counts.append(sample_count)

    smartfl_result = smartfl_base.optimize_smartfl_weights(
        global_model,
        local_states,
        sample_counts,
        selected_clients,
        task_batches,
        args.server_lr_cap,
        args.server_opt_steps,
        args.server_opt_weight_tol,
        args.grad_damping,
        device,
    )
    next_state = smartfl_base.aggregate_states_on_cpu(local_states, smartfl_result["alpha"], global_model)
    global_model.load_state_dict(next_state)

    task_loss, task_acc = smartfl_base.evaluate(global_model, task_eval_loader, device)
    gen_loss, gen_acc = smartfl_base.evaluate(global_model, gen_test_loader, device)

    return {
        "device": device,
        "global_model": global_model,
        "selected_clients": selected_clients,
        "sample_counts": sample_counts,
        "smartfl_weights": smartfl_result["alpha"],
        "proxy_loss": smartfl_result["proxy_loss"],
        "server_opt_steps_used": smartfl_result["steps_used"],
        "server_opt_early_stopped": smartfl_result["early_stopped"],
        "server_opt_stop_reason": smartfl_result["stop_reason"],
        "server_opt_final_change_l1": smartfl_result["final_step_change_l1"],
        "server_opt_final_change_max_abs": smartfl_result["final_step_change_max_abs"],
        "task_loss": task_loss,
        "task_acc": task_acc,
        "gen_loss": gen_loss,
        "gen_acc": gen_acc,
    }


def run_federated_learning(args):
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be <= num_clients")

    set_seed(FIXED_SEED)
    devices = smartfl_base.resolve_devices(args)
    device = devices[0]

    full_dataset, full_targets = load_yahooanswers(args)
    private_indices, task_indices, gen_test_indices, split_metadata = balanced_three_way_split_indices(
        full_targets,
        args.private_train_size,
        args.task_size,
        args.gen_test_size,
        FIXED_SEED,
    )

    private_targets = full_targets[private_indices]
    partition_positions = smartfl_base.iid_partition(private_targets, args.num_clients, FIXED_SEED)
    client_indices = smartfl_base.build_client_indices(private_indices, partition_positions)

    train_loader_kwargs = smartfl_base.build_loader_kwargs(args, shuffle=True)
    eval_loader_kwargs = smartfl_base.build_loader_kwargs(args, shuffle=False)
    client_loader_kwargs = dict(train_loader_kwargs)
    client_loader_kwargs["num_workers"] = 0
    client_loader_kwargs.pop("persistent_workers", None)

    client_loaders, client_sizes = smartfl_base.build_client_loaders(
        full_dataset, client_indices, args.batch_size, client_loader_kwargs, FIXED_SEED
    )

    task_subset = Subset(full_dataset, task_indices.tolist())
    server_proxy_batch_size = len(task_subset)
    server_proxy_batches_per_round = 1
    task_loader = DataLoader(task_subset, batch_size=server_proxy_batch_size, **train_loader_kwargs)
    task_iterator = iter(task_loader)
    task_eval_loader = DataLoader(task_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    gen_test_subset = Subset(full_dataset, gen_test_indices.tolist())
    gen_test_loader = DataLoader(gen_test_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    print(
        "Min-gap-LR setup:",
        f"code_version={CODE_VERSION}",
        "dataset=YahooAnswersTopics",
        "model=TextCNN",
        "partition=IID",
        "splits=private/task/gen_test",
        f"split_strategy={split_metadata['split_strategy']}",
        "server_weight_optimization=enabled",
        "effective_lr=sum_i(grad_i-min_grad)",
        f"grad_damping={args.grad_damping}",
        f"grad_stability_mean_threshold={args.grad_stability_mean_threshold}",
        f"grad_stability_range_threshold={args.grad_stability_range_threshold}",
        f"server_lr_cap={args.server_lr_cap}",
        f"clients={args.num_clients}",
        f"clients_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"private_train_size={len(private_indices)}",
        f"task_size={len(task_indices)}",
        f"gen_test_size={len(gen_test_indices)}",
        f"server_proxy_batch_size={server_proxy_batch_size}",
        f"server_proxy_batches_per_round={server_proxy_batches_per_round}",
        f"labels={YAHOO_ANSWERS_LABELS}",
        f"fixed_seed={FIXED_SEED}",
        f"device={device}",
    )

    global_model = build_model(args).to(device)
    client_weight_sums = [0.0 for _ in range(args.num_clients)]
    client_selected_counts = [0 for _ in range(args.num_clients)]

    for round_idx in range(1, args.rounds + 1):
        selected_clients = smartfl_base.random.sample(range(args.num_clients), args.clients_per_round)
        print(f"Round {round_idx:03d} selected_clients={selected_clients}")
        task_batches, task_iterator = smartfl_base.next_batches(
            task_loader,
            task_iterator,
            server_proxy_batches_per_round,
        )
        global_snapshot = smartfl_base.clone_state_dict_to_cpu(global_model.state_dict())

        trial_model = build_model(args).to(device)
        trial_model.load_state_dict(global_snapshot)
        round_result = run_round_on_device(
            round_idx,
            device,
            args,
            selected_clients,
            client_loaders,
            client_sizes,
            trial_model,
            task_batches,
            task_eval_loader,
            gen_test_loader,
        )
        global_model = round_result["global_model"]

        for client_id, weight in zip(round_result["selected_clients"], round_result["smartfl_weights"]):
            client_weight_sums[client_id] += float(weight)
            client_selected_counts[client_id] += 1

        print(
            f"Round {round_idx:03d}:",
            f"selected_clients={selected_clients}",
            f"proxy_loss={round_result['proxy_loss']:.4f}",
            f"server_steps_used={round_result['server_opt_steps_used']}",
            f"server_early_stopped={round_result['server_opt_early_stopped']}",
            f"server_stop_reason={round_result['server_opt_stop_reason']}",
            f"task_loss={round_result['task_loss']:.4f}",
            f"task_acc={round_result['task_acc']:.4f}",
            f"gen_loss={round_result['gen_loss']:.4f}",
            f"gen_acc={round_result['gen_acc']:.4f}",
        )

    output_path = smartfl_base.resolve_output_path(args.output_json)
    client_records = []
    for client_id in range(args.num_clients):
        selected_count = client_selected_counts[client_id]
        weight_sum = client_weight_sums[client_id]
        client_records.append(
            {
                "client_id": client_id,
                "selected_count": selected_count,
                "weight_sum": weight_sum,
                "average_weight_when_selected": (weight_sum / selected_count) if selected_count > 0 else None,
                "average_weight_over_all_rounds": weight_sum / args.rounds,
            }
        )

    smartfl_base.write_json_snapshot(
        {
            "code_version": CODE_VERSION,
            "dataset": "YahooAnswersTopics",
            "model": "TextCNN",
            "partition": "IID",
            "labels": YAHOO_ANSWERS_LABELS,
            "split_strategy": split_metadata["split_strategy"],
            "requested_private_train_size": args.private_train_size,
            "requested_task_size": args.task_size,
            "requested_gen_test_size": args.gen_test_size,
            "balanced_split_class_counts": split_metadata["class_counts"],
            "rounds": args.rounds,
            "num_clients": args.num_clients,
            "clients_per_round": args.clients_per_round,
            "client_size_unique_values": sorted(set(client_sizes)),
            "client_size_histogram": {
                str(size): sum(1 for client_size in client_sizes if client_size == size)
                for size in sorted(set(client_sizes))
            },
            "iid_note": (
                "Class proportions are IID because each class pool is shuffled and split across all clients. "
                "Client sample counts are not exactly equal under the current np.array_split implementation."
            ),
            "client_average_weights": client_records,
        },
        output_path,
    )
    print(f"Client average weights JSON saved to: {output_path}")

    return global_model


def main():
    args = parse_args()
    run_federated_learning(args)


if __name__ == "__main__":
    main()
