from pathlib import Path

import federated_speechcommands_hist_mingaplr_base as hist_base
import federated_speechcommands_smartfl_mingaplr_compare as base
from federated_speechcommands_hist_mingaplr_dirichlet_fixed_noise_common import (
    RATE_LABELS,
    build_fixed_noise_client_loaders,
    get_client_noise_rate,
)


CODE_VERSION = "speechcommands_mingaplr_fixed_noise_hist_compare_v1"
DEFAULT_HIST_JSON = Path("outputs/speechcommands_mingaplr_fixed_noise_hist_compare/results.json")


def build_iid_fixed_noise_setup(full_dataset, full_targets, private_indices, private_targets, args, client_loader_kwargs):
    partition_positions = base.smartfl_base.iid_partition(private_targets, args.num_clients, base.FIXED_SEED)
    client_indices = base.smartfl_base.build_client_indices(private_indices, partition_positions)
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
            "labels": base.audio_base.COMMAND_LABELS,
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
