from pathlib import Path

import federated_cifar10_mingaplr_compare as base
import federated_cifar10_hist_mingaplr_base as hist_base
from federated_cifar10_hist_mingaplr_dirichlet_fixed_noise_common import (
    build_dirichlet_fixed_noise_setup,
)


CODE_VERSION = "mingaplr_dirichlet1_window_noise_hist_compare_v6"
DEFAULT_HIST_JSON = Path("outputs/mingaplr_dirichlet1_fixed_noise_hist_compare/results17.json")


def main():
    args = base.parse_args()
    if args.output_json == base.DEFAULT_CLIENT_AVG_WEIGHT_JSON:
        args.output_json = DEFAULT_HIST_JSON
    hist_base.run_hist_experiment(
        args=args,
        code_version=CODE_VERSION,
        partition_name="Dirichlet",
        output_json=args.output_json,
        setup_fn=build_dirichlet_fixed_noise_setup(1.0),
    )


if __name__ == "__main__":
    main()
