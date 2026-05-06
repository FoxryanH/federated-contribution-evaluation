from pathlib import Path

import controlled_time_analysis_common as common


CODE_VERSION = "lcefl_controlled_time_analysis_s1_v1"
DEFAULT_OUTPUT_JSON = Path("outputs/time_analysis/lcefl/s1_time_summary.json")


def main():
    common.run_time_analysis(
        base_script_name="federated_cifar10_lcefl_mingaplr_S1_controlled_compare.py",
        method_tag="lcefl",
        code_version=CODE_VERSION,
        default_output_json=DEFAULT_OUTPUT_JSON,
        description=(
            "Controlled time-analysis wrapper for LCEFL S1. "
            "Runs participant counts 3-7 with 10 FL rounds and 1000 samples per client."
        ),
    )


if __name__ == "__main__":
    main()
