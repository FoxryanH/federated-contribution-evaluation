from collections import OrderedDict
from pathlib import Path

import torch

from federated_cifar10_hist_mingaplr_dirichlet05_window_noise_attack_common import (
    AttackSpec,
    assert_state_is_finite,
    parse_args,
    run_experiment,
)


CODE_VERSION = "mingaplr_dirichlet05_window_noise_sign_flipping_contribution_compare_v1"
DEFAULT_OUTPUT_JSON = Path("outputs/mingaplr_dirichlet05_window_noise_sign_flipping_compare/results1.json")


def add_attack_args(parser):
    parser.add_argument(
        "--sign-flip-gamma",
        type=float,
        default=1.0,
        help="Attack strength gamma in attacked_update = -gamma * clean_update",
    )


def validate_attack_args(args):
    if args.sign_flip_gamma < 0:
        raise ValueError("sign_flip_gamma must be non-negative")


def attack_config(args):
    return {"sign_flip_gamma": float(args.sign_flip_gamma)}


def apply_sign_flipping_attack(global_state, local_state, args, shared_state, client_id, round_idx):
    del client_id, round_idx
    attacked_state = OrderedDict()
    parameter_names = shared_state["parameter_names"]
    for key, local_value in local_state.items():
        if key in parameter_names and torch.is_tensor(local_value) and local_value.is_floating_point():
            global_value = global_state[key]
            delta = local_value - global_value
            attacked_state[key] = (global_value - args.sign_flip_gamma * delta).detach().clone().cpu()
        elif torch.is_tensor(local_value):
            attacked_state[key] = local_value.detach().clone().cpu()
        else:
            attacked_state[key] = local_value
    assert_state_is_finite(attacked_state, "sign_flipping_attacked_state")
    return attacked_state


ATTACK_SPEC = AttackSpec(
    attack_type="sign_flipping",
    display_name="SignFlip",
    code_version=CODE_VERSION,
    default_output_json=DEFAULT_OUTPUT_JSON,
    description=(
        "CIFAR-10 contribution evaluation under Dirichlet-0.5, fixed label noise, "
        "and sign-flipping Byzantine attack: Ours plus accuracy-only FedAvg baseline"
    ),
    add_attack_args=add_attack_args,
    validate_attack_args=validate_attack_args,
    attack_config=attack_config,
    apply_attack=apply_sign_flipping_attack,
)


def main():
    args = parse_args(ATTACK_SPEC)
    run_experiment(args, ATTACK_SPEC)


if __name__ == "__main__":
    main()
