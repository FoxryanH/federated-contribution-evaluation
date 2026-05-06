from collections import OrderedDict
from pathlib import Path

import torch

from federated_cifar10_hist_mingaplr_dirichlet05_window_noise_attack_common import (
    AttackSpec,
    assert_state_is_finite,
    parse_args,
    run_experiment,
)


CODE_VERSION = "mingaplr_dirichlet05_window_noise_scaling_contribution_compare_v1"
DEFAULT_OUTPUT_JSON = Path("outputs/mingaplr_dirichlet05_window_noise_scaling_compare/results0.5-1.json")


def add_attack_args(parser):
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=10.0,
        help="Scaling factor in attacked_update = scaling_factor * clean_update around the global model",
    )


def validate_attack_args(args):
    if args.scaling_factor < 0:
        raise ValueError("scaling_factor must be non-negative")


def attack_config(args):
    return {"scaling_factor": float(args.scaling_factor)}


def apply_scaling_attack(global_state, local_state, args, shared_state, client_id, round_idx):
    del client_id, round_idx
    attacked_state = OrderedDict()
    parameter_names = shared_state["parameter_names"]
    for key, local_value in local_state.items():
        if key in parameter_names and torch.is_tensor(local_value) and local_value.is_floating_point():
            global_value = global_state[key]
            delta = local_value - global_value
            attacked_state[key] = (global_value + args.scaling_factor * delta).detach().clone().cpu()
        elif torch.is_tensor(local_value):
            attacked_state[key] = local_value.detach().clone().cpu()
        else:
            attacked_state[key] = local_value
    assert_state_is_finite(attacked_state, "scaling_attacked_state")
    return attacked_state


ATTACK_SPEC = AttackSpec(
    attack_type="scaling",
    display_name="Scaling",
    code_version=CODE_VERSION,
    default_output_json=DEFAULT_OUTPUT_JSON,
    description=(
        "CIFAR-10 contribution evaluation under Dirichlet-0.5, fixed label noise, "
        "and scaling Byzantine attack: Ours plus accuracy-only FedAvg baseline"
    ),
    add_attack_args=add_attack_args,
    validate_attack_args=validate_attack_args,
    attack_config=attack_config,
    apply_attack=apply_scaling_attack,
)


def main():
    args = parse_args(ATTACK_SPEC)
    run_experiment(args, ATTACK_SPEC)


if __name__ == "__main__":
    main()
