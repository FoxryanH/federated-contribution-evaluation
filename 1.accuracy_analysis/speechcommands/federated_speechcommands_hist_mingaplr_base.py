from pathlib import Path

import torch

import federated_speechcommands_smartfl_mingaplr_compare as base


WINDOW_SIZE = 10
PREFERRED_CATEGORY_ORDER = ["clean", "noise20", "noise40", "noise60", "iid_shared_distribution"]


def optimize_smartfl_weights_with_analysis(
    model,
    local_states,
    sample_counts,
    selected_clients,
    proxy_batches,
    server_lr_cap,
    server_opt_steps,
    server_opt_weight_tol,
    grad_damping,
    device,
):
    del server_opt_weight_tol

    model = model.to(device)
    model.eval()

    alpha = torch.tensor(sample_counts, dtype=torch.float32, device=device)
    alpha = alpha / alpha.sum()
    base_alpha = alpha.detach().cpu().tolist()
    grad_sum = torch.zeros_like(alpha)
    grad_steps = 0

    final_change_l1 = 0.0
    final_change_max_abs = 0.0
    steps_used = 0
    early_stopped = False
    stop_reason = "max_steps"
    proxy_loss_history = []
    alpha_history = [alpha.detach().clone()]
    mean_abs_grad_history = []

    for step_idx in range(server_opt_steps):
        alpha_var = alpha.detach().clone().requires_grad_(True)
        proxy_loss_value, proxy_grad = base.smartfl_base.compute_proxy_loss_and_grad_for_alpha(
            model, local_states, alpha_var, proxy_batches, device
        )
        with torch.no_grad():
            proxy_loss_history.append(float(proxy_loss_value))
            grad_sum = grad_sum + proxy_grad.detach()
            grad_steps += 1

            alpha_before = alpha_var.detach().clone()
            min_grad = torch.min(proxy_grad)
            max_grad = torch.max(proxy_grad)
            grad_range = float((max_grad - min_grad).item())
            mean_abs_grad = float(torch.mean(torch.abs(proxy_grad)).item())
            mean_abs_grad_history.append(mean_abs_grad)
            grad_gap = proxy_grad - min_grad
            uncapped_server_lr = float(torch.sum(grad_gap).item())
            if server_lr_cap > 0:
                effective_server_lr = min(uncapped_server_lr, float(server_lr_cap))
            else:
                effective_server_lr = uncapped_server_lr

            grad_scale = torch.sum(torch.abs(proxy_grad))
            if not torch.isfinite(grad_scale) or grad_scale < 1e-12:
                scaled_grad = torch.zeros_like(proxy_grad)
            else:
                scaled_grad = proxy_grad / (grad_scale + float(grad_damping))

            raw = alpha_var * torch.exp(-effective_server_lr * scaled_grad)
            next_alpha = base.smartfl_base.normalize_raw_weights(raw)
            delta = next_alpha - alpha
            final_change_l1 = float(delta.abs().sum().item())
            final_change_max_abs = float(delta.abs().max().item())

            print(
                f"  Server opt step {step_idx + 1:02d}: "
                f"proxy_loss={proxy_loss_value:.6f} "
                f"min_grad={float(min_grad.item()):.6f} "
                f"max_grad={float(max_grad.item()):.6f} "
                f"grad_range={grad_range:.6f} "
                f"mean_abs_grad={mean_abs_grad:.6f} "
                f"gap_sum_lr={uncapped_server_lr:.6f} "
                f"effective_lr={effective_server_lr:.6f} "
                f"grad_damping={float(grad_damping):.6f} "
                f"change_l1={final_change_l1:.6f} "
                f"change_max_abs={final_change_max_abs:.6f}"
            )
            for local_idx, client_id in enumerate(selected_clients):
                print(
                    f"    Client {client_id}: "
                    f"alpha_before={float(alpha_before[local_idx].item()):.6f} "
                    f"grad={float(proxy_grad[local_idx].item()):+.6f} "
                    f"grad_gap={float(grad_gap[local_idx].item()):.6f} "
                    f"scaled_grad={float(scaled_grad[local_idx].item()):+.6f} "
                    f"raw={float(raw[local_idx].item()):.6f} "
                    f"alpha_after={float(next_alpha[local_idx].item()):.6f}"
                )

            if mean_abs_grad < 1.0 and grad_range < 1.0:
                steps_used = step_idx
                early_stopped = True
                stop_reason = "grad_stability_revert"
                final_change_l1 = 0.0
                final_change_max_abs = 0.0
                print(
                    "    Early stop before applying update: "
                    "mean_abs_grad and grad_range both fell below 1.0. "
                    f"mean_abs_grad={mean_abs_grad:.6f}, grad_range={grad_range:.6f}"
                )
                break

            if len(proxy_loss_history) >= 4:
                recent_losses = proxy_loss_history[-4:]
                if recent_losses[0] < recent_losses[1] < recent_losses[2] < recent_losses[3]:
                    alpha = alpha_history[-4].detach().clone()
                    steps_used = step_idx + 1
                    early_stopped = True
                    stop_reason = "loss_rise_revert"
                    final_change_l1 = 0.0
                    final_change_max_abs = 0.0
                    print(
                        "    Early stop and rollback: proxy_loss rose for three consecutive steps. "
                        f"recent_proxy_losses={[round(loss, 6) for loss in recent_losses]}"
                    )
                    break

            alpha = next_alpha
            alpha_history.append(alpha.detach().clone())
            steps_used = step_idx + 1

    with torch.no_grad():
        proxy_loss = base.smartfl_base.evaluate_proxy_loss_for_alpha(
            model, local_states, alpha, proxy_batches, device
        )

    mean_gradients = (grad_sum / max(grad_steps, 1)).detach().cpu().tolist()
    first_step_mean_abs_grad = mean_abs_grad_history[0] if mean_abs_grad_history else 0.0
    mean_abs_grad_over_optimization = (
        sum(mean_abs_grad_history) / len(mean_abs_grad_history) if mean_abs_grad_history else 0.0
    )
    return {
        "alpha": alpha.detach().cpu().tolist(),
        "base_alpha": base_alpha,
        "proxy_loss": float(proxy_loss.item()),
        "steps_used": steps_used,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "final_step_change_l1": final_change_l1,
        "final_step_change_max_abs": final_change_max_abs,
        "mean_gradients": mean_gradients,
        "first_step_mean_abs_grad": first_step_mean_abs_grad,
        "mean_abs_grad_over_optimization": mean_abs_grad_over_optimization,
    }


def compute_leave_one_out_proxy_metrics(model, local_states, analysis_alpha, proxy_batches, device):
    alpha_tensor = torch.tensor(analysis_alpha, dtype=torch.float32, device=device)
    with torch.no_grad():
        full_proxy_loss = float(
            base.smartfl_base.evaluate_proxy_loss_for_alpha(model, local_states, alpha_tensor, proxy_batches, device)
            .item()
        )

    loo_losses = []
    for client_idx in range(alpha_tensor.numel()):
        loo_alpha = alpha_tensor.clone()
        loo_alpha[client_idx] = 0.0
        loo_alpha = base.smartfl_base.normalize_raw_weights(loo_alpha)
        with torch.no_grad():
            loo_loss = float(
                base.smartfl_base.evaluate_proxy_loss_for_alpha(
                    model, local_states, loo_alpha, proxy_batches, device
                ).item()
            )
        loo_losses.append(loo_loss)

    return full_proxy_loss, loo_losses


def build_client_analysis_records(
    selected_clients,
    sample_counts,
    optimized_weights,
    mean_gradients,
    loo_full_proxy_loss,
    loo_losses,
    client_metadata,
):
    total_samples = float(sum(sample_counts))
    base_weights = [sample_count / total_samples for sample_count in sample_counts]
    client_records = []

    for local_idx, client_id in enumerate(selected_clients):
        client_record = {
            "client_id": client_id,
            "base_weight": float(base_weights[local_idx]),
            "optimized_weight": float(optimized_weights[local_idx]),
            "mean_weight_derivative": float(mean_gradients[local_idx]),
            "loo_proxy_loss": float(loo_losses[local_idx]),
            "loo_proxy_loss_delta": float(loo_losses[local_idx] - loo_full_proxy_loss),
        }
        client_record.update(client_metadata[client_id])
        client_records.append(client_record)

    return client_records


def run_round_on_device_with_analysis(
    round_idx,
    device,
    args,
    selected_clients,
    client_loaders,
    client_sizes,
    client_metadata,
    global_model,
    task_batches,
    task_eval_loader,
    gen_test_loader,
):
    global_model = global_model.to(device)
    global_state = base.smartfl_base.clone_state_dict_to_cpu(global_model.state_dict())

    local_states = []
    sample_counts = []
    for client_id in selected_clients:
        local_seed = base.FIXED_SEED + round_idx * 1000 + client_id
        local_state, sample_count = base.client_local_train(
            global_state,
            client_loaders[client_id],
            client_sizes[client_id],
            device,
            args,
            local_seed,
        )
        local_states.append(local_state)
        sample_counts.append(sample_count)

    smartfl_result = optimize_smartfl_weights_with_analysis(
        global_model,
        local_states,
        sample_counts,
        selected_clients,
        task_batches,
        args.server_lr_cap,
        args.server_opt_steps,
        args.server_opt_weight_tol,
        args.grad_damping,
        device,
    )
    proxy_loss_post_optimization = smartfl_result["proxy_loss"]

    loo_full_proxy_loss, loo_losses = compute_leave_one_out_proxy_metrics(
        global_model,
        local_states,
        smartfl_result["base_alpha"],
        task_batches,
        device,
    )

    next_state = base.smartfl_base.aggregate_states_on_cpu(local_states, smartfl_result["alpha"], global_model)
    global_model.load_state_dict(next_state)

    task_loss, task_acc = base.smartfl_base.evaluate(global_model, task_eval_loader, device)
    gen_loss, gen_acc = base.smartfl_base.evaluate(global_model, gen_test_loader, device)

    client_records = build_client_analysis_records(
        selected_clients,
        sample_counts,
        smartfl_result["alpha"],
        smartfl_result["mean_gradients"],
        loo_full_proxy_loss,
        loo_losses,
        client_metadata,
    )

    return {
        "global_model": global_model,
        "round_record": {
            "round_index": round_idx,
            "selected_clients": selected_clients,
            "proxy_loss_post_optimization": proxy_loss_post_optimization,
            "task_loss_post_optimization": task_loss,
            "task_acc_post_optimization": task_acc,
            "gen_loss_post_optimization": gen_loss,
            "gen_acc_post_optimization": gen_acc,
            "server_steps_used": smartfl_result["steps_used"],
            "server_early_stopped": smartfl_result["early_stopped"],
            "server_stop_reason": smartfl_result["stop_reason"],
            "server_final_change_l1": smartfl_result["final_step_change_l1"],
            "server_final_change_max_abs": smartfl_result["final_step_change_max_abs"],
            "first_step_mean_abs_grad": smartfl_result["first_step_mean_abs_grad"],
            "mean_abs_grad_over_optimization": smartfl_result["mean_abs_grad_over_optimization"],
            "clients": client_records,
        },
    }


def print_setup_banner(code_version, partition_name, args, private_indices, task_indices, gen_test_indices, device, startup_tokens):
    print(
        "Min-gap-LR hist-input setup:",
        f"code_version={code_version}",
        f"base_code_version={base.CODE_VERSION}",
        "dataset=SpeechCommands",
        "model=AudioCNN",
        f"partition={partition_name}",
        "splits=private/task/gen_test",
        "server_weight_optimization=enabled",
        "effective_lr=sum_i(grad_i-min_grad)",
        "analysis=mean_grad_and_leave_one_out_proxy_loss",
        f"grad_damping={args.grad_damping}",
        f"server_lr_cap={args.server_lr_cap}",
        f"clients={args.num_clients}",
        f"clients_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"private_train_size={len(private_indices)}",
        f"task_size={len(task_indices)}",
        f"gen_test_size={len(gen_test_indices)}",
        f"labels={base.audio_base.COMMAND_LABELS}",
        f"fixed_seed={base.FIXED_SEED}",
        f"device={device}",
        *startup_tokens,
    )


def build_window_average_weights(round_records, window_size):
    if not round_records:
        return []

    encountered_categories = []
    for round_record in round_records:
        for client_record in round_record["clients"]:
            category = client_record.get("client_category", "unknown")
            if category not in encountered_categories:
                encountered_categories.append(category)

    category_order = [category for category in PREFERRED_CATEGORY_ORDER if category in encountered_categories]
    category_order.extend([category for category in encountered_categories if category not in PREFERRED_CATEGORY_ORDER])

    window_summaries = []
    for offset in range(0, len(round_records), window_size):
        window_rounds = round_records[offset:offset + window_size]
        if not window_rounds:
            continue

        category_stats = {category: {"weight_sum": 0.0, "selected_count": 0} for category in category_order}
        for round_record in window_rounds:
            for client_record in round_record["clients"]:
                category = client_record.get("client_category", "unknown")
                category_stats[category]["weight_sum"] += float(client_record["optimized_weight"])
                category_stats[category]["selected_count"] += 1

        average_weights = {}
        for category in category_order:
            stats = category_stats[category]
            if stats["selected_count"] == 0:
                average_weights[category] = None
            else:
                average_weights[category] = stats["weight_sum"] / stats["selected_count"]

        window_summaries.append(
            {
                "window_start_round": window_rounds[0]["round_index"],
                "window_end_round": window_rounds[-1]["round_index"],
                "average_weights": average_weights,
            }
        )

    return window_summaries


def build_round_gradient_metrics(round_records):
    return [
        {
            "round_index": round_record["round_index"],
            "selected_clients": round_record["selected_clients"],
            "first_step_mean_abs_grad": round_record["first_step_mean_abs_grad"],
            "mean_abs_grad_over_optimization": round_record["mean_abs_grad_over_optimization"],
        }
        for round_record in round_records
    ]


def build_client_average_weights(round_records, num_clients, total_rounds, client_metadata):
    client_stats = {client_id: {"weight_sum": 0.0, "selected_count": 0} for client_id in range(num_clients)}

    for round_record in round_records:
        for client_record in round_record["clients"]:
            client_id = int(client_record["client_id"])
            client_stats[client_id]["weight_sum"] += float(client_record["optimized_weight"])
            client_stats[client_id]["selected_count"] += 1

    client_average_weights = []
    for client_id in range(num_clients):
        stats = client_stats[client_id]
        selected_count = stats["selected_count"]
        weight_sum = stats["weight_sum"]
        client_average_weights.append(
            {
                "client_id": client_id,
                **client_metadata[client_id],
                "selected_count": selected_count,
                "weight_sum": weight_sum,
                "average_weight_when_selected": (weight_sum / selected_count if selected_count > 0 else None),
                "average_weight_over_all_rounds": (weight_sum / total_rounds if total_rounds > 0 else None),
            }
        )

    return client_average_weights


def run_hist_experiment(args, code_version, partition_name, output_json, setup_fn):
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be <= num_clients")

    base.set_seed(base.FIXED_SEED)
    devices = base.smartfl_base.resolve_devices(args)
    device = devices[0]

    full_dataset, full_targets = base.load_speech_commands(args)
    private_indices, task_indices, gen_test_indices = base.smartfl_base.stratified_three_way_split_indices(
        full_targets,
        args.task_size,
        args.gen_test_size,
        base.FIXED_SEED,
    )
    private_targets = full_targets[private_indices]

    train_loader_kwargs = base.smartfl_base.build_loader_kwargs(args, shuffle=True)
    eval_loader_kwargs = base.smartfl_base.build_loader_kwargs(args, shuffle=False)
    client_loader_kwargs = dict(train_loader_kwargs)
    client_loader_kwargs["num_workers"] = 0
    client_loader_kwargs.pop("persistent_workers", None)

    setup = setup_fn(
        full_dataset=full_dataset,
        full_targets=full_targets,
        private_indices=private_indices,
        private_targets=private_targets,
        args=args,
        client_loader_kwargs=client_loader_kwargs,
    )

    client_loaders = setup["client_loaders"]
    client_sizes = setup["client_sizes"]
    client_metadata = setup["client_metadata"]
    startup_tokens = setup.get("startup_tokens", [])
    config_extras = setup.get("config_extras", {})
    selected_clients_info_fn = setup.get("selected_clients_info_fn")

    task_subset = base.Subset(full_dataset, task_indices.tolist())
    server_proxy_batch_size = len(task_subset)
    server_proxy_batches_per_round = 1
    task_loader = base.DataLoader(task_subset, batch_size=server_proxy_batch_size, **train_loader_kwargs)
    task_iterator = iter(task_loader)
    task_eval_loader = base.DataLoader(task_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    gen_test_subset = base.Subset(full_dataset, gen_test_indices.tolist())
    gen_test_loader = base.DataLoader(gen_test_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    startup_tokens = startup_tokens + [
        f"server_proxy_batch_size={server_proxy_batch_size}",
        f"server_proxy_batches_per_round={server_proxy_batches_per_round}",
    ]
    print_setup_banner(
        code_version,
        partition_name,
        args,
        private_indices,
        task_indices,
        gen_test_indices,
        device,
        startup_tokens,
    )

    global_model = base.build_model().to(device)
    round_records = []

    for round_idx in range(1, args.rounds + 1):
        selected_clients = base.smartfl_base.random.sample(range(args.num_clients), args.clients_per_round)
        selected_clients_suffix = ""
        if selected_clients_info_fn is not None:
            extra_info = selected_clients_info_fn(selected_clients, client_metadata)
            if extra_info:
                selected_clients_suffix = f" {extra_info}"
        print(f"Round {round_idx:03d} selected_clients={selected_clients}{selected_clients_suffix}")

        task_batches, task_iterator = base.smartfl_base.next_batches(
            task_loader, task_iterator, server_proxy_batches_per_round
        )
        global_snapshot = base.smartfl_base.clone_state_dict_to_cpu(global_model.state_dict())

        trial_model = base.build_model().to(device)
        trial_model.load_state_dict(global_snapshot)
        round_result = run_round_on_device_with_analysis(
            round_idx,
            device,
            args,
            selected_clients,
            client_loaders,
            client_sizes,
            client_metadata,
            trial_model,
            task_batches,
            task_eval_loader,
            gen_test_loader,
        )
        global_model = round_result["global_model"]
        round_records.append(round_result["round_record"])

        print(
            f"Round {round_idx:03d}:",
            f"selected_clients={selected_clients}",
            f"proxy_loss={round_result['round_record']['proxy_loss_post_optimization']:.4f}",
            f"server_steps_used={round_result['round_record']['server_steps_used']}",
            f"server_early_stopped={round_result['round_record']['server_early_stopped']}",
            f"server_stop_reason={round_result['round_record']['server_stop_reason']}",
            f"task_loss={round_result['round_record']['task_loss_post_optimization']:.4f}",
            f"task_acc={round_result['round_record']['task_acc_post_optimization']:.4f}",
            f"gen_loss={round_result['round_record']['gen_loss_post_optimization']:.4f}",
            f"gen_acc={round_result['round_record']['gen_acc_post_optimization']:.4f}",
        )

    payload = {
        "config": {
            "code_version": code_version,
            "base_code_version": base.CODE_VERSION,
            "dataset": "SpeechCommands",
            "model": "AudioCNN",
            "labels": base.audio_base.COMMAND_LABELS,
            "partition": partition_name,
            "num_clients": args.num_clients,
            "clients_per_round": args.clients_per_round,
            "rounds": args.rounds,
            "task_size": len(task_indices),
            "gen_test_size": len(gen_test_indices),
            "private_train_size": len(private_indices),
            "fixed_seed": base.FIXED_SEED,
            "window_size": WINDOW_SIZE,
            "grad_damping": args.grad_damping,
            "server_lr_cap": args.server_lr_cap,
            "server_opt_steps": args.server_opt_steps,
            **config_extras,
        },
        "rounds": round_records,
        "round_gradient_metrics": build_round_gradient_metrics(round_records),
        "window_average_weights": build_window_average_weights(round_records, WINDOW_SIZE),
        "client_average_weights": build_client_average_weights(
            round_records, args.num_clients, args.rounds, client_metadata
        ),
    }
    output_path = base.smartfl_base.resolve_output_path(output_json)
    base.smartfl_base.write_json_snapshot(payload, output_path)
    print(f"Hist-input JSON saved to: {output_path}")


def build_fixed_noise_selected_clients_info(selected_clients, client_metadata):
    selected_noise_info = []
    for client_id in selected_clients:
        noise_rate = client_metadata[client_id].get("noise_rate", 0.0)
        if noise_rate > 0.0:
            selected_noise_info.append(f"{client_id}:{int(round(100 * noise_rate))}%")
    if not selected_noise_info:
        return ""
    return f"fixed_noise_clients={selected_noise_info}"
