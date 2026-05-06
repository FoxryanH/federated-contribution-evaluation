from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from federated_cifar10_hist_mingaplr_dirichlet05_window_noise_attack_common import (
    AttackSpec,
    assert_state_is_finite,
    parse_args,
    run_experiment,
)


CODE_VERSION = "mingaplr_dirichlet05_window_noise_gaussian_noise_contribution_compare_v1"
DEFAULT_OUTPUT_JSON = Path("outputs/mingaplr_dirichlet05_window_noise_gaussian_noise_compare/results1.json")


def add_attack_args(parser):
    parser.add_argument(
        "--gaussian-noise-scale",
        type=float,
        default=60.0,
        help="Noise scale multiplying the per-tensor std of the clean client update delta",
    )


def validate_attack_args(args):
    if args.gaussian_noise_scale < 0:
        raise ValueError("gaussian_noise_scale must be non-negative")


def attack_config(args):
    return {"gaussian_noise_scale": float(args.gaussian_noise_scale)}


def apply_gaussian_noise_attack(global_state, local_state, args, shared_state, client_id, round_idx):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(10_000_000 + round_idx * 10_000 + client_id))

    attacked_state = OrderedDict()
    parameter_names = shared_state["parameter_names"]
    for key, local_value in local_state.items():
        if key in parameter_names and torch.is_tensor(local_value) and local_value.is_floating_point():
            global_value = global_state[key]
            delta = (local_value - global_value).detach().clone().cpu()
            delta_std = float(delta.std(unbiased=False).item()) if delta.numel() > 1 else float(delta.abs().mean().item())
            if not np.isfinite(delta_std) or delta_std <= 0.0:
                noise = torch.zeros_like(delta)
            else:
                noise = torch.randn(delta.shape, generator=generator, dtype=delta.dtype) * (
                    args.gaussian_noise_scale * delta_std
                )
            attacked_state[key] = (global_value + delta + noise).detach().clone().cpu()
        elif torch.is_tensor(local_value):
            attacked_state[key] = local_value.detach().clone().cpu()
        else:
            attacked_state[key] = local_value
    assert_state_is_finite(attacked_state, "gaussian_noise_attacked_state")
    return attacked_state


ATTACK_SPEC = AttackSpec(
    attack_type="gaussian_noise",
    display_name="GaussianNoise",
    code_version=CODE_VERSION,
    default_output_json=DEFAULT_OUTPUT_JSON,
    description=(
        "CIFAR-10 contribution evaluation under Dirichlet-0.5, fixed label noise, "
        "and Gaussian-noise Byzantine attack: Ours plus accuracy-only FedAvg baseline"
    ),
    add_attack_args=add_attack_args,
    validate_attack_args=validate_attack_args,
    attack_config=attack_config,
    apply_attack=apply_gaussian_noise_attack,
)


def main():
    args = parse_args(ATTACK_SPEC)
    run_experiment(args, ATTACK_SPEC)


if __name__ == "__main__":
    main()
