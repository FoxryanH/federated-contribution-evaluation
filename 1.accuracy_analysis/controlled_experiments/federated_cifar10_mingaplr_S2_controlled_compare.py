import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SIBLING_CIFAR10_DIR = REPO_ROOT / "1.accuracy_analysis" / "cifar10"
for extra_path in (REPO_ROOT, SIBLING_CIFAR10_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

import federated_cifar10_mingaplr_S1_controlled_compare as s1_base


CODE_VERSION = "mingaplr_controlled_compare_s2_default_v1"
DEFAULT_OUTPUT_JSON = Path("outputs/mingaplr_controlled_compare/s2_results.json")
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
    s1_base.run_federated_learning(args)


if __name__ == "__main__":
    main()
