import numpy as np
import torch
from torch.utils.data import Dataset

import federated_cifar10_mingaplr_compare as base
import federated_cifar10_hist_mingaplr_base as hist_base


RATE_LABELS = {
    0.0: "clean",
    0.2: "noise20",
    0.4: "noise40",
    0.6: "noise60",
}


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


def equal_size_dirichlet_partition(private_indices, private_targets, num_clients, alpha, seed):
    private_indices = np.asarray(private_indices, dtype=np.int64)
    private_targets = np.asarray(private_targets, dtype=np.int64)
    rng = np.random.default_rng(seed)

    total_private = len(private_indices)
    per_client_size = total_private // num_clients
    usable_total = per_client_size * num_clients

    if usable_total == 0:
        raise ValueError("Not enough private samples to allocate to clients")

    class_to_indices = {}
    for class_id in np.unique(private_targets):
        class_pool = private_indices[private_targets == class_id].copy()
        rng.shuffle(class_pool)
        class_to_indices[int(class_id)] = class_pool.tolist()

    class_ids = sorted(class_to_indices.keys())
    client_indices = [[] for _ in range(num_clients)]

    for client_id in range(num_clients):
        class_pref = rng.dirichlet(np.full(len(class_ids), alpha, dtype=np.float64))
        tentative = np.floor(class_pref * per_client_size).astype(int)
        remainder = per_client_size - int(tentative.sum())
        if remainder > 0:
            order = np.argsort(-class_pref)
            for offset in range(remainder):
                tentative[order[offset % len(order)]] += 1

        allocated = []
        for idx, class_id in enumerate(class_ids):
            take = tentative[idx]
            available = len(class_to_indices[class_id])
            actual_take = min(take, available)
            if actual_take > 0:
                allocated.extend(class_to_indices[class_id][:actual_take])
                del class_to_indices[class_id][:actual_take]

        while len(allocated) < per_client_size:
            available_classes = [class_id for class_id in class_ids if class_to_indices[class_id]]
            if not available_classes:
                raise RuntimeError("Ran out of class samples during equal-size Dirichlet partition")
            sampled_class = int(rng.choice(available_classes))
            allocated.append(class_to_indices[sampled_class].pop())

        rng.shuffle(allocated)
        client_indices[client_id] = allocated

    unused_private_indices = []
    for class_id in class_ids:
        unused_private_indices.extend(class_to_indices[class_id])
    rng.shuffle(unused_private_indices)

    client_sizes = [len(indices) for indices in client_indices]
    if len(set(client_sizes)) != 1:
        raise RuntimeError(f"Client sizes are not equal: {sorted(set(client_sizes))}")

    return client_indices, unused_private_indices, per_client_size


def get_client_noise_rate(client_id):
    if 20 <= client_id <= 29:
        return 0.2
    if 30 <= client_id <= 39:
        return 0.4
    if 40 <= client_id <= 49:
        return 0.6
    return 0.0


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


def build_fixed_noise_client_loaders(dataset, full_targets, client_indices, batch_size, loader_kwargs, base_seed):
    client_loaders = []
    client_sizes = []
    full_targets = np.asarray(full_targets, dtype=np.int64)

    for client_id, indices in enumerate(client_indices):
        noise_rate = get_client_noise_rate(client_id)
        noisy_labels = build_fixed_noisy_labels(
            full_targets[np.asarray(indices, dtype=np.int64)],
            noise_rate,
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


def build_dirichlet_fixed_noise_setup(dirichlet_alpha):
    def setup_fn(full_dataset, full_targets, private_indices, private_targets, args, client_loader_kwargs):
        client_indices, unused_private_indices, per_client_private_size = equal_size_dirichlet_partition(
            private_indices,
            private_targets,
            args.num_clients,
            dirichlet_alpha,
            base.FIXED_SEED,
        )
        client_loaders, client_sizes = build_fixed_noise_client_loaders(
            full_dataset, full_targets, client_indices, args.batch_size, client_loader_kwargs, base.FIXED_SEED
        )

        client_metadata = []
        for client_id in range(args.num_clients):
            noise_rate = get_client_noise_rate(client_id)
            client_metadata.append(
                {
                    "client_category": RATE_LABELS[noise_rate],
                    "noise_rate": noise_rate,
                }
            )

        return {
            "client_loaders": client_loaders,
            "client_sizes": client_sizes,
            "client_metadata": client_metadata,
            "startup_tokens": [
                f"dirichlet_alpha={dirichlet_alpha}",
                "client_private_size=equal",
                "client_noise=fixed_at_initialization",
                "noise_groups=0-19:0%,20-29:20%,30-39:40%,40-49:60%",
                "noise_flip_scope=within_client_existing_classes_only",
                f"per_client_private_size={per_client_private_size}",
                f"unused_private_samples={len(unused_private_indices)}",
            ],
            "config_extras": {
                "dirichlet_alpha": dirichlet_alpha,
                "client_private_size": "equal",
                "per_client_private_size": per_client_private_size,
                "unused_private_samples": len(unused_private_indices),
                "noise_groups": {
                    "0-19": "clean",
                    "20-29": "noise20",
                    "30-39": "noise40",
                    "40-49": "noise60",
                },
            },
            "selected_clients_info_fn": hist_base.build_fixed_noise_selected_clients_info,
        }

    return setup_fn
