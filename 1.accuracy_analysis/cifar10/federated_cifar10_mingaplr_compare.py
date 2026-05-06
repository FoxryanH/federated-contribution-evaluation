import argparse
import gc
import importlib.util
import json
import os
import random
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_resnet18_builder():
    module_path = REPO_ROOT / "model" / "resnet-18.py"
    spec = importlib.util.spec_from_file_location("model_resnet18", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ResNet18 definition from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResNet18


RESNET18_BUILDER = load_resnet18_builder()
FIXED_SEED = 42
CODE_VERSION = "mingaplr_compare_v6_damped_no_early_stop"
DEFAULT_GRAD_DAMPING = 5
DEFAULT_CLIENT_AVG_WEIGHT_JSON = Path("outputs/mingaplr_compare/client_average_weights17.json")


def build_model():
    return RESNET18_BUILDER()


def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 SmartFL baseline with min-gap-driven server learning rate"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_CLIENT_AVG_WEIGHT_JSON)
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
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def resolve_output_path(output_path: Path) -> Path:
    if output_path.is_absolute():
        return output_path
    return Path(__file__).resolve().parent / output_path


def write_json_snapshot(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def parse_gpu_ids(gpu_ids):
    return [int(token.strip()) for token in gpu_ids.split(",") if token.strip()]


def query_gpu_metrics(gpu_ids):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    except Exception:
        return {}

    metrics = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
            if index not in gpu_ids:
                continue
            util = float(parts[1])
            memory_free = float(parts[2])
            memory_total = float(parts[3])
        except ValueError:
            continue
        metrics[index] = {
            "util": util,
            "memory_free": memory_free,
            "memory_total": memory_total,
            "memory_free_ratio": (memory_free / memory_total) if memory_total > 0 else 0.0,
        }
    return metrics


def resolve_devices(args):
    if not torch.cuda.is_available():
        return [torch.device("cpu")]

    visible_gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not visible_gpu_ids:
        visible_gpu_ids = list(range(torch.cuda.device_count()))

    valid_gpu_ids = [gpu_id for gpu_id in visible_gpu_ids if 0 <= gpu_id < torch.cuda.device_count()]
    if not valid_gpu_ids:
        return [torch.device("cpu")]

    metrics = query_gpu_metrics(valid_gpu_ids)
    if metrics:
        valid_gpu_ids = sorted(
            valid_gpu_ids,
            key=lambda gpu_id: (
                metrics.get(gpu_id, {}).get("util", float("inf")),
                -metrics.get(gpu_id, {}).get("memory_free_ratio", 0.0),
                -metrics.get(gpu_id, {}).get("memory_free", 0.0),
                gpu_id,
            ),
        )

    return [torch.device(f"cuda:{gpu_id}") for gpu_id in valid_gpu_ids]


def build_loader_kwargs(args, shuffle):
    kwargs = {
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def load_cifar10(data_dir):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def stratified_three_way_split_indices(targets, task_size, gen_test_size, seed):
    targets = np.asarray(targets, dtype=np.int64)
    classes = np.unique(targets)
    rng = np.random.default_rng(seed)

    private_indices = []
    task_indices = []
    gen_test_indices = []

    task_base = task_size // len(classes)
    task_remainder = task_size % len(classes)
    test_base = gen_test_size // len(classes)
    test_remainder = gen_test_size % len(classes)

    for offset, class_id in enumerate(classes):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        task_take = task_base + (1 if offset < task_remainder else 0)
        test_take = test_base + (1 if offset < test_remainder else 0)
        if task_take + test_take >= len(class_indices):
            raise ValueError("task_size/gen_test_size are too large for stratified splitting")

        task_indices.extend(class_indices[:task_take].tolist())
        gen_test_indices.extend(class_indices[task_take:task_take + test_take].tolist())
        private_indices.extend(class_indices[task_take + test_take:].tolist())

    rng.shuffle(private_indices)
    rng.shuffle(task_indices)
    rng.shuffle(gen_test_indices)
    return (
        np.asarray(private_indices, dtype=np.int64),
        np.asarray(task_indices, dtype=np.int64),
        np.asarray(gen_test_indices, dtype=np.int64),
    )


def iid_partition(private_targets, num_clients, seed):
    private_targets = np.asarray(private_targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    partition_positions = [[] for _ in range(num_clients)]

    for class_id in np.unique(private_targets):
        class_positions = np.where(private_targets == class_id)[0]
        rng.shuffle(class_positions)
        splits = np.array_split(class_positions, num_clients)
        for client_id, split in enumerate(splits):
            partition_positions[client_id].extend(split.tolist())

    for positions in partition_positions:
        rng.shuffle(positions)
    return partition_positions


def build_client_indices(full_indices, partition_positions):
    return [full_indices[np.asarray(positions, dtype=np.int64)].tolist() for positions in partition_positions]


def build_client_loaders(dataset, client_indices, batch_size, loader_kwargs, base_seed):
    client_loaders = []
    client_sizes = []
    for client_id, indices in enumerate(client_indices):
        subset = Subset(dataset, indices)
        generator = torch.Generator()
        generator.manual_seed(base_seed + client_id)
        client_loader_kwargs = dict(loader_kwargs)
        client_loader_kwargs["generator"] = generator
        client_loaders.append(DataLoader(subset, batch_size=batch_size, **client_loader_kwargs))
        client_sizes.append(len(indices))
    return client_loaders, client_sizes


def clone_state_dict_to_cpu(state_dict):
    return OrderedDict((key, value.detach().clone().cpu()) for key, value in state_dict.items())


def get_param_names(model):
    return [name for name, _ in model.named_parameters()]


def get_buffer_names(model):
    return [name for name, _ in model.named_buffers()]


def build_optimizer(model, args):
    if args.local_optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.local_lr,
            momentum=args.local_momentum,
            weight_decay=args.local_weight_decay,
        )
    return torch.optim.Adam(model.parameters(), lr=args.local_lr, weight_decay=args.local_weight_decay)


def clear_device_cache(device):
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def client_local_train(global_state, dataset_loader, sample_count, device, args, local_seed):
    local_model = build_model().to(device)
    local_model.load_state_dict(global_state)
    local_model.train()

    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)

    optimizer = build_optimizer(local_model, args)
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

    local_state = clone_state_dict_to_cpu(local_model.state_dict())
    del optimizer
    del criterion
    del local_model
    clear_device_cache(device)
    return local_state, sample_count


def next_batch(data_loader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(data_loader)
        batch = next(iterator)
    return batch, iterator


def next_batches(data_loader, iterator, num_batches):
    batches = []
    for _ in range(num_batches):
        batch, iterator = next_batch(data_loader, iterator)
        batches.append(batch)
    return batches, iterator


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            total_loss += F.cross_entropy(outputs, targets, reduction="sum").item()
            total_correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)
    return total_loss / total_samples, total_correct / total_samples


def build_param_and_buffer_state(model, state_dict, device):
    params = OrderedDict()
    buffers = OrderedDict()
    for name in get_param_names(model):
        params[name] = state_dict[name].to(device)
    for name in get_buffer_names(model):
        buffers[name] = state_dict[name].to(device)
    return params, buffers


def aggregate_states_on_cpu(state_dicts, weights, model):
    total_weight = float(sum(weights))
    param_names = set(get_param_names(model))
    buffer_names = set(get_buffer_names(model))
    aggregated = OrderedDict()

    for key in state_dicts[0].keys():
        reference = state_dicts[0][key]
        if key in param_names or (key in buffer_names and reference.is_floating_point()):
            value = torch.zeros_like(reference)
            for state_dict, weight in zip(state_dicts, weights):
                value = value + state_dict[key] * (weight / total_weight)
            aggregated[key] = value
        else:
            aggregated[key] = reference.clone()
    return aggregated


def aggregate_state_for_alpha(model, local_states, alpha, device):
    param_names = set(get_param_names(model))
    buffer_names = set(get_buffer_names(model))
    aggregated = OrderedDict()

    for key in local_states[0].keys():
        reference = local_states[0][key]
        if key in param_names:
            stacked = torch.stack([state[key].to(device, non_blocking=True) for state in local_states], dim=0)
            view_shape = [alpha.shape[0]] + [1] * (stacked.dim() - 1)
            aggregated[key] = torch.sum(alpha.view(*view_shape) * stacked, dim=0)
        elif key in buffer_names and reference.is_floating_point():
            stacked = torch.stack([state[key].to(device, non_blocking=True) for state in local_states], dim=0)
            view_shape = [alpha.shape[0]] + [1] * (stacked.dim() - 1)
            aggregated[key] = torch.sum(alpha.detach().view(*view_shape) * stacked, dim=0)
        else:
            aggregated[key] = reference.to(device, non_blocking=True) if torch.is_tensor(reference) else reference
    return aggregated


def compute_proxy_loss_and_grad_for_alpha(model, local_states, alpha, proxy_batches, device):
    total_loss = 0.0
    total_grad = torch.zeros_like(alpha)
    for inputs, targets in proxy_batches:
        aggregated_state = aggregate_state_for_alpha(model, local_states, alpha, device)
        params, buffers = build_param_and_buffer_state(model, aggregated_state, device)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = functional_call(model, (params, buffers), (inputs,))
        batch_loss = F.cross_entropy(outputs, targets)
        batch_grad = torch.autograd.grad(batch_loss, alpha)[0]
        total_loss += float(batch_loss.detach().item())
        total_grad = total_grad + batch_grad.detach()
        del aggregated_state, params, buffers, outputs, batch_loss
    num_batches = len(proxy_batches)
    return total_loss / num_batches, total_grad / num_batches


def evaluate_proxy_loss_for_alpha(model, local_states, alpha, proxy_batches, device):
    aggregated_state = aggregate_state_for_alpha(model, local_states, alpha, device)
    params, buffers = build_param_and_buffer_state(model, aggregated_state, device)
    total_loss = 0.0
    for inputs, targets in proxy_batches:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = functional_call(model, (params, buffers), (inputs,))
        total_loss += F.cross_entropy(outputs, targets)
    return total_loss / len(proxy_batches)


def normalize_raw_weights(vector):
    denominator = torch.sum(vector)
    if not torch.isfinite(denominator) or torch.abs(denominator) < 1e-12:
        return torch.full_like(vector, 1.0 / vector.numel())
    return vector / denominator


def optimize_smartfl_weights(
    model,
    local_states,
    sample_counts,
    selected_clients,
    proxy_batches,
    server_lr_cap,
    server_opt_steps,
    server_opt_weight_tol,
    grad_damping,
    device,
):
    model = model.to(device)
    model.eval()
    del server_opt_weight_tol

    alpha = torch.tensor(sample_counts, dtype=torch.float32, device=device)
    alpha = alpha / alpha.sum()
    final_change_l1 = 0.0
    final_change_max_abs = 0.0
    steps_used = 0
    stop_reason = "max_steps"

    for step_idx in range(server_opt_steps):
        alpha_var = alpha.detach().clone().requires_grad_(True)
        proxy_loss_value, proxy_grad = compute_proxy_loss_and_grad_for_alpha(
            model, local_states, alpha_var, proxy_batches, device
        )
        with torch.no_grad():
            alpha_before = alpha_var.detach().clone()
            min_grad = torch.min(proxy_grad)
            max_grad = torch.max(proxy_grad)
            grad_range = float((max_grad - min_grad).item())
            mean_abs_grad = float(torch.mean(torch.abs(proxy_grad)).item())
            grad_gap = proxy_grad - min_grad
            uncapped_server_lr = float(torch.sum(grad_gap).item())
            if server_lr_cap > 0:
                effective_server_lr = min(uncapped_server_lr, float(server_lr_cap))
            else:
                effective_server_lr = uncapped_server_lr

            grad_scale = torch.sum(torch.abs(proxy_grad))
            if not torch.isfinite(grad_scale):
                scaled_grad = torch.zeros_like(proxy_grad)
            else:
                scaled_grad = proxy_grad / (grad_scale + float(grad_damping))

            raw = alpha_var * torch.exp(-effective_server_lr * scaled_grad)
            next_alpha = normalize_raw_weights(raw)
            delta = next_alpha - alpha
            final_change_l1 = float(delta.abs().sum().item())
            final_change_max_abs = float(delta.abs().max().item())

            print(
                f"  Server opt step {step_idx + 1:02d}: "
                f"proxy_loss={proxy_loss_value:.6f} "
                f"min_grad={float(min_grad.item()):.6f} "
                f"max_grad={float(max_grad.item()):.6f} "
                f"grad_range={grad_range:.6f} "
                f"mean_abs_grad={mean_abs_grad:.6f} "
                f"grad_damping={float(grad_damping):.6f} "
                f"gap_sum_lr={uncapped_server_lr:.6f} "
                f"effective_lr={effective_server_lr:.6f} "
                f"change_l1={final_change_l1:.6f} "
                f"change_max_abs={final_change_max_abs:.6f}"
            )
            for local_idx, client_id in enumerate(selected_clients):
                print(
                    f"    Client {client_id}: "
                    f"alpha_before={float(alpha_before[local_idx].item()):.6f} "
                    f"grad={float(proxy_grad[local_idx].item()):+.6f} "
                    f"grad_gap={float(grad_gap[local_idx].item()):.6f} "
                    f"scaled_grad={float(scaled_grad[local_idx].item()):+.6f} "
                    f"raw={float(raw[local_idx].item()):.6f} "
                    f"alpha_after={float(next_alpha[local_idx].item()):.6f}"
                )

            alpha = next_alpha
            steps_used = step_idx + 1

    with torch.no_grad():
        proxy_loss = evaluate_proxy_loss_for_alpha(model, local_states, alpha, proxy_batches, device)

    return {
        "alpha": alpha.detach().cpu().tolist(),
        "proxy_loss": float(proxy_loss.item()),
        "steps_used": steps_used,
        "early_stopped": False,
        "stop_reason": stop_reason,
        "final_step_change_l1": final_change_l1,
        "final_step_change_max_abs": final_change_max_abs,
    }


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
    global_state = clone_state_dict_to_cpu(global_model.state_dict())

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

    smartfl_result = optimize_smartfl_weights(
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
    next_state = aggregate_states_on_cpu(local_states, smartfl_result["alpha"], global_model)
    global_model.load_state_dict(next_state)

    task_loss, task_acc = evaluate(global_model, task_eval_loader, device)
    gen_loss, gen_acc = evaluate(global_model, gen_test_loader, device)

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
    devices = resolve_devices(args)
    device = devices[0]

    train_dataset, test_dataset = load_cifar10(args.data_dir)
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    full_targets = np.asarray(train_dataset.targets + test_dataset.targets, dtype=np.int64)
    private_indices, task_indices, gen_test_indices = stratified_three_way_split_indices(
        full_targets,
        args.task_size,
        args.gen_test_size,
        FIXED_SEED,
    )

    private_targets = full_targets[private_indices]
    partition_positions = iid_partition(private_targets, args.num_clients, FIXED_SEED)
    client_indices = build_client_indices(private_indices, partition_positions)

    train_loader_kwargs = build_loader_kwargs(args, shuffle=True)
    eval_loader_kwargs = build_loader_kwargs(args, shuffle=False)
    client_loader_kwargs = dict(train_loader_kwargs)
    client_loader_kwargs["num_workers"] = 0
    client_loader_kwargs.pop("persistent_workers", None)

    client_loaders, client_sizes = build_client_loaders(
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
        "dataset=CIFAR-10",
        "partition=IID",
        "splits=private/task/gen_test",
        "server_weight_optimization=enabled",
        "effective_lr=sum_i(grad_i-min_grad)",
        f"grad_damping={args.grad_damping}",
        f"server_lr_cap={args.server_lr_cap}",
        f"clients={args.num_clients}",
        f"clients_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"private_train_size={len(private_indices)}",
        f"task_size={len(task_indices)}",
        f"gen_test_size={len(gen_test_indices)}",
        f"server_proxy_batch_size={server_proxy_batch_size}",
        f"server_proxy_batches_per_round={server_proxy_batches_per_round}",
        f"fixed_seed={FIXED_SEED}",
        f"device={device}",
    )

    global_model = build_model().to(device)
    client_weight_sums = [0.0 for _ in range(args.num_clients)]
    client_selected_counts = [0 for _ in range(args.num_clients)]

    for round_idx in range(1, args.rounds + 1):
        selected_clients = random.sample(range(args.num_clients), args.clients_per_round)
        print(f"Round {round_idx:03d} selected_clients={selected_clients}")
        task_batches, task_iterator = next_batches(task_loader, task_iterator, server_proxy_batches_per_round)
        global_snapshot = clone_state_dict_to_cpu(global_model.state_dict())

        trial_model = build_model().to(device)
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

    output_path = resolve_output_path(args.output_json)
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

    write_json_snapshot(
        {
            "code_version": CODE_VERSION,
            "partition": "IID",
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
