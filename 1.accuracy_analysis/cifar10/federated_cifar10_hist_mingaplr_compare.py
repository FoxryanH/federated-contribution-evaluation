from pathlib import Path

import federated_cifar10_mingaplr_compare as base
import federated_cifar10_hist_mingaplr_base as hist_base


CODE_VERSION = "mingaplr_hist_compare_v6"
DEFAULT_HIST_JSON = Path("outputs/mingaplr_hist_compare/results17.json")


def build_iid_setup(full_dataset, full_targets, private_indices, private_targets, args, client_loader_kwargs):
    del full_targets

    partition_positions = base.iid_partition(private_targets, args.num_clients, base.FIXED_SEED)
    client_indices = base.build_client_indices(private_indices, partition_positions)
    client_loaders, client_sizes = base.build_client_loaders(
        full_dataset, client_indices, args.batch_size, client_loader_kwargs, base.FIXED_SEED
    )

    client_metadata = [{"client_category": "iid_shared_distribution"} for _ in range(args.num_clients)]
    return {
        "client_loaders": client_loaders,
        "client_sizes": client_sizes,
        "client_metadata": client_metadata,
        "startup_tokens": [
            "client_distribution=iid_shared_distribution",
        ],
        "config_extras": {
            "client_size_unique_values": sorted(set(client_sizes)),
            "client_size_histogram": {
                str(size): sum(1 for client_size in client_sizes if client_size == size)
                for size in sorted(set(client_sizes))
            },
        },
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
        setup_fn=build_iid_setup,
    )


if __name__ == "__main__":
    main()
