import sys
from pathlib import Path

import federated_cifar10_loo_mingaplr_S1_controlled_compare as s1_base


CODE_VERSION = "loo_controlled_compare_s2_global_v2_adaptive_shrinkage"
DEFAULT_OUTPUT_JSON = Path("outputs/loo_controlled_compare/s2_global_results.json")
DEFAULT_CLIENT_NOISE_RATES = "0,10,20,30,40,50"


def _has_cli_flag(flag_name):
    return flag_name in sys.argv[1:]


def parse_args():
    args = s1_base.parse_args()
    if not _has_cli_flag("--scenario-tag"):
        args.scenario_tag = "S2"
    if not _has_cli_flag("--output-json"):
        args.output_json = DEFAULT_OUTPUT_JSON
    if not _has_cli_flag("--client-noise-rates"):
        args.client_noise_rates = DEFAULT_CLIENT_NOISE_RATES
    return args


def main():
    s1_base.CODE_VERSION = CODE_VERSION
    args = parse_args()
    s1_base.run_loo_experiment(args)


if __name__ == "__main__":
    main()
