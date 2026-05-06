from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import federated_cifar10_mingaplr_compare as base
import federated_cifar10_hist_mingaplr_base as hist_base


CODE_VERSION = "mingaplr_fixed_noise_hist_compare_v6"
DEFAULT_HIST_JSON = Path("outputs/mingaplr_fixed_noise_hist_compare/results17.json")
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


def build_iid_fixed_noise_setup(full_dataset, full_targets, private_indices, private_targets, args, client_loader_kwargs):
    partition_positions = base.iid_partition(private_targets, args.num_clients, base.FIXED_SEED)
    client_indices = base.build_client_indices(private_indices, partition_positions)
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
            "client_noise=fixed_at_initialization",
            "noise_groups=0-19:0%,20-29:20%,30-39:40%,40-49:60%",
            "noise_flip_scope=within_client_existing_classes_only",
        ],
        "config_extras": {
            "noise_groups": {
                "0-19": "clean",
                "20-29": "noise20",
                "30-39": "noise40",
                "40-49": "noise60",
            },
            "client_size_unique_values": sorted(set(client_sizes)),
            "client_size_histogram": {
                str(size): sum(1 for client_size in client_sizes if client_size == size)
                for size in sorted(set(client_sizes))
            },
        },
        "selected_clients_info_fn": hist_base.build_fixed_noise_selected_clients_info,
    }


def main():
    args = base.parse_args()
    if args.output_json == base.DEFAULT_CLIENT_AVG_WEIGHT_JSON:
        args.output_json = DEFAULT_HIST_JSON
    hist_base.run_hist_experiment(
        args=args,
        code_version=CODE_VERSION,
        partition_name="IID",
        output_json=args.output_json,
        setup_fn=build_iid_fixed_noise_setup,
    )


if __name__ == "__main__":
    main()
