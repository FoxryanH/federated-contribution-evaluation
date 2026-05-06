import argparse
import json
from pathlib import Path


EPS = 1e-8
PRIOR_MASS = 0.1
WINDOW_SIZE = 10
DEFAULT_T0 = 3
DEFAULT_HISTORY_PRIOR_COUNT = 0
PREFERRED_CATEGORY_ORDER = ["clean", "noise20", "noise40", "noise60"]
QUALITY_BY_CATEGORY = {
    "iid_shared_distribution": 1.0,
    "clean": 1.0,
    "noise20": 0.8,
    "noise40": 0.6,
    "noise60": 0.4,
    "attack": 0.0,
}


def mean(values):
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def infer_actual_quality(item):
    noise_rate = item.get("noise_rate")
    if noise_rate is not None:
        return 1.0 - float(noise_rate)

    category = item.get("client_category", "")
    if category in QUALITY_BY_CATEGORY:
        return QUALITY_BY_CATEGORY[category]
    if isinstance(category, str) and category.startswith("noise"):
        suffix = category[5:]
        if suffix.isdigit():
            return 1.0 - float(suffix) / 100.0
    return None


def scale_secondary_signal_to_primary_range(primary_values, secondary_values):
    primary_max = max((float(value) for value in primary_values), default=0.0)
    secondary_max = max((float(value) for value in secondary_values), default=0.0)
    if secondary_max <= primary_max + EPS:
        return [float(value) for value in secondary_values]
    scale = primary_max / max(secondary_max, EPS)
    return [float(value) * scale for value in secondary_values]


def pearson_correlation(x_values, y_values):
    if len(x_values) < 2:
        return None
    mean_x = mean(x_values)
    mean_y = mean(y_values)
    centered_x = [x - mean_x for x in x_values]
    centered_y = [y - mean_y for y in y_values]
    var_x = sum(x * x for x in centered_x)
    var_y = sum(y * y for y in centered_y)
    if var_x <= EPS or var_y <= EPS:
        return None
    covariance = sum(x * y for x, y in zip(centered_x, centered_y))
    return covariance / ((var_x ** 0.5) * (var_y ** 0.5))


def average_ranks(values):
    indexed_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed_values):
        end = idx + 1
        while end < len(indexed_values) and indexed_values[end][1] == indexed_values[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for position in range(idx, end):
            ranks[indexed_values[position][0]] = avg_rank
        idx = end
    return ranks


def build_monotonicity_metrics(client_summaries):
    valid_clients = []
    for client_summary in client_summaries:
        contribution = client_summary.get("average_normalized_contribution")
        quality = client_summary.get("actual_quality")
        if contribution is None or quality is None:
            continue
        valid_clients.append(
            {
                "client_id": client_summary["client_id"],
                "contribution": float(contribution),
                "quality": float(quality),
            }
        )

    if len(valid_clients) < 2:
        return {
            "quality_definition": "1-noise_rate",
            "num_valid_clients": len(valid_clients),
            "pearson_correlation": None,
            "spearman_correlation": None,
            "monotonicity_ratio": None,
            "satisfied_pairs": 0,
            "total_pairs": 0,
        }

    contributions = [item["contribution"] for item in valid_clients]
    qualities = [item["quality"] for item in valid_clients]

    satisfied_pairs = 0
    total_pairs = 0
    for left_idx in range(len(valid_clients)):
        for right_idx in range(left_idx + 1, len(valid_clients)):
            total_pairs += 1
            left = valid_clients[left_idx]
            right = valid_clients[right_idx]
            if abs(left["quality"] - right["quality"]) <= EPS:
                satisfied_pairs += 1
            elif left["quality"] > right["quality"]:
                if left["contribution"] >= right["contribution"] - EPS:
                    satisfied_pairs += 1
            else:
                if right["contribution"] >= left["contribution"] - EPS:
                    satisfied_pairs += 1

    return {
        "quality_definition": "1-noise_rate",
        "num_valid_clients": len(valid_clients),
        "pearson_correlation": pearson_correlation(contributions, qualities),
        "spearman_correlation": pearson_correlation(
            average_ranks(contributions),
            average_ranks(qualities),
        ),
        "monotonicity_ratio": (satisfied_pairs / total_pairs) if total_pairs > 0 else None,
        "satisfied_pairs": satisfied_pairs,
        "total_pairs": total_pairs,
    }


def build_window_average_contributions(records, window_size):
    if not records:
        return []

    encountered_categories = []
    for record in records:
        category = record["client_category"]
        if category not in encountered_categories:
            encountered_categories.append(category)

    category_order = [category for category in PREFERRED_CATEGORY_ORDER if category in encountered_categories]
    category_order.extend(
        [category for category in encountered_categories if category not in PREFERRED_CATEGORY_ORDER]
    )

    rounds_to_records = {}
    for record in records:
        rounds_to_records.setdefault(record["round_index"], []).append(record)

    sorted_round_indices = sorted(rounds_to_records.keys())
    window_summaries = []

    for offset in range(0, len(sorted_round_indices), window_size):
        window_rounds = sorted_round_indices[offset:offset + window_size]
        if not window_rounds:
            continue

        category_stats = {
            category: {"sum": 0.0, "count": 0}
            for category in category_order
        }
        for round_index in window_rounds:
            for record in rounds_to_records[round_index]:
                category = record["client_category"]
                category_stats[category]["sum"] += record["normalized_contribution"]
                category_stats[category]["count"] += 1

        average_contributions = {}
        for category in category_order:
            stats = category_stats[category]
            if stats["count"] == 0:
                average_contributions[category] = None
            else:
                average_contributions[category] = stats["sum"] / stats["count"]

        window_summaries.append(
            {
                "window_start_round": window_rounds[0],
                "window_end_round": window_rounds[-1],
                "average_normalized_contribution": average_contributions,
            }
        )

    return window_summaries


def build_client_average_contributions(records):
    if not records:
        return []

    client_stats = {}
    for record in records:
        client_id = record["client_id"]
        if client_id not in client_stats:
            client_stats[client_id] = {
                "client_category": record["client_category"],
                "normalized_sum": 0.0,
                "count": 0,
            }
        client_stats[client_id]["normalized_sum"] += record["normalized_contribution"]
        client_stats[client_id]["count"] += 1

    client_summaries = []
    for client_id in sorted(client_stats.keys()):
        stats = client_stats[client_id]
        client_summaries.append(
            {
                "client_id": client_id,
                "client_category": stats["client_category"],
                "selected_count": stats["count"],
                "actual_quality": infer_actual_quality(stats),
                "average_normalized_contribution": stats["normalized_sum"] / max(stats["count"], 1),
            }
        )

    return client_summaries


def derive_output_json_path(input_json_path, output_json_path=None):
    input_json_path = Path(input_json_path)
    if output_json_path is not None:
        return Path(output_json_path)
    return input_json_path.with_name(f"{input_json_path.stem}_attack{input_json_path.suffix}")


def resolve_num_clients(data):
    config = data.get("config")
    if isinstance(config, dict) and config.get("num_clients") is not None:
        return int(config["num_clients"])

    if data.get("num_clients") is not None:
        return int(data["num_clients"])

    client_average_weights = data.get("client_average_weights")
    if isinstance(client_average_weights, list) and client_average_weights:
        client_ids = [int(item["client_id"]) for item in client_average_weights if item.get("client_id") is not None]
        if client_ids:
            return max(client_ids) + 1

    round_items = resolve_round_items(data)
    max_client_id = None
    for round_item in round_items:
        for client_item in round_item.get("clients", []):
            client_id = client_item.get("client_id")
            if client_id is None:
                continue
            client_id = int(client_id)
            max_client_id = client_id if max_client_id is None else max(max_client_id, client_id)

    if max_client_id is not None:
        return max_client_id + 1

    raise KeyError("Unable to resolve num_clients from input JSON")


def resolve_round_items(data):
    round_items = data.get("rounds")
    if isinstance(round_items, list) and round_items and isinstance(round_items[0], dict) and "clients" in round_items[0]:
        return round_items

    round_items = data.get("rounds_detail")
    if isinstance(round_items, list):
        return round_items

    raise KeyError("Unable to resolve round records from input JSON. Expected 'rounds' or 'rounds_detail'.")


def build_subjective_logic(input_json_path, output_json_path, t0, history_prior_count):
    input_json_path = Path(input_json_path)
    output_json_path = Path(output_json_path)

    with input_json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    num_clients = resolve_num_clients(data)
    round_items = resolve_round_items(data)
    hist_weight_sum = {cid: 0.0 for cid in range(num_clients)}
    hist_weight_count = {cid: 0 for cid in range(num_clients)}
    output_records = []

    for round_item in round_items:
        t = round_item["round_index"]
        clients = round_item["clients"]
        if not clients:
            continue

        round_contributions = []
        contribution_shrink = min(t / max(t0, EPS), 1.0)

        raw_weight_gains = []
        raw_loo_pos_values = []
        raw_weight_drops = []
        raw_loo_neg_values = []
        raw_loo_deltas = []

        for client_item in clients:
            w_opt = client_item["optimized_weight"]
            loo_delta = client_item["loo_proxy_loss_delta"]
            raw_weight_gains.append(max(w_opt - client_item["base_weight"], 0.0))
            raw_loo_pos_values.append(max(loo_delta, 0.0))
            raw_weight_drops.append(max(client_item["base_weight"] - client_item["optimized_weight"], 0.0))
            raw_loo_neg_values.append(max(-loo_delta, 0.0))
            raw_loo_deltas.append(loo_delta)

        scaled_loo_pos_values = scale_secondary_signal_to_primary_range(
            raw_weight_gains,
            raw_loo_pos_values,
        )
        scaled_loo_neg_values = scale_secondary_signal_to_primary_range(
            raw_weight_drops,
            raw_loo_neg_values,
        )

        for client_item, weight_gain, loo_delta, loo_pos_i, w_drop, loo_neg_i in zip(
            clients,
            raw_weight_gains,
            raw_loo_deltas,
            scaled_loo_pos_values,
            raw_weight_drops,
            scaled_loo_neg_values,
        ):
            cid = client_item["client_id"]

            p1 = weight_gain
            p2 = loo_pos_i
            r_round = (p1 + p2) / 2.0

            n1 = w_drop
            n2 = loo_neg_i
            s_round = (n1 + n2) / 2.0

            denom = r_round + s_round + PRIOR_MASS
            belief = r_round / denom
            disbelief = s_round / denom
            uncertainty = PRIOR_MASS / denom

            base_weight = float(client_item["base_weight"])
            if hist_weight_count.get(cid, 0) > 0:
                historical_avg_weight = (
                    history_prior_count * base_weight + hist_weight_sum[cid]
                ) / (history_prior_count + hist_weight_count[cid])
            else:
                historical_avg_weight = base_weight

            uncertainty_weight_prior = 0.5 * base_weight + 0.5 * historical_avg_weight
            expected_contribution_raw = belief + uncertainty_weight_prior * uncertainty
            expected_contribution = base_weight + contribution_shrink * (expected_contribution_raw - base_weight)
            conservative_contribution = expected_contribution * (1.0 - uncertainty)
            history_weight_update_value = client_item["optimized_weight"]

            round_contributions.append(expected_contribution)

            output_records.append(
                {
                    "round_index": t,
                    "client_id": cid,
                    "client_category": client_item.get("client_category", "unknown"),
                    "base_weight": client_item["base_weight"],
                    "optimized_weight": client_item["optimized_weight"],
                    "weight_drop": w_drop,
                    "loo_proxy_loss": client_item["loo_proxy_loss"],
                    "loo_proxy_loss_delta": loo_delta,
                    "p1_weight_adv": p1,
                    "p2_loo_pos_adv": p2,
                    "n1_weight_drop_bad": n1,
                    "n2_loo_neg_bad": n2,
                    "round_positive_evidence": r_round,
                    "round_negative_evidence": s_round,
                    "round_reliability": 1.0,
                    "belief": belief,
                    "disbelief": disbelief,
                    "uncertainty": uncertainty,
                    "historical_avg_weight": historical_avg_weight,
                    "uncertainty_weight_prior": uncertainty_weight_prior,
                    "expected_contribution_raw": expected_contribution_raw,
                    "contribution_shrink_T0": t0,
                    "contribution_shrink_factor": contribution_shrink,
                    "history_weight_update_value": history_weight_update_value,
                    "expected_contribution": expected_contribution,
                    "conservative_contribution": conservative_contribution,
                }
            )

        for client_item in clients:
            cid = client_item["client_id"]
            history_weight_update_value = float(client_item["optimized_weight"])
            hist_weight_sum[cid] += history_weight_update_value
            hist_weight_count[cid] += 1

        start_idx = len(output_records) - len(clients)
        total_contribution = sum(round_contributions) + EPS
        for record_idx in range(start_idx, len(output_records)):
            output_records[record_idx]["normalized_contribution"] = (
                output_records[record_idx]["expected_contribution"] / total_contribution
            )

    client_average_contributions = build_client_average_contributions(output_records)

    result = {
        "subjective_logic_config": {
            "prior_mass": PRIOR_MASS,
            "window_size": WINDOW_SIZE,
            "window_summary_value": "normalized_contribution",
            "client_summary_value": "normalized_contribution",
            "history_prior_count": history_prior_count,
            "contribution_shrink_T0": t0,
            "normalization": "raw_values",
            "positive_evidence": [
                "weight_gain(raw)",
                "positive_loo_delta(raw)",
            ],
            "negative_evidence": [
                "weight_drop(raw)",
                "negative_loo_delta(raw)",
            ],
            "uncertainty": "prior_mass_only",
            "history_update": "bayesian_shrinkage_of_selected_client_optimized_weight_toward_base_weight",
            "expected_contribution_shrink": "base_weight + min(round_index / T0, 1) * (expected_contribution_raw - base_weight)",
            "uncertainty_prior": "0.5*base_weight + 0.5*bayesian_shrunk_historical_average_optimized_weight",
        },
        "records": output_records,
        "window_average_contributions": build_window_average_contributions(output_records, WINDOW_SIZE),
        "client_average_contributions": client_average_contributions,
        "monotonicity_metrics": build_monotonicity_metrics(client_average_contributions),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build subjective-logic contribution records from a round-wise experiment results.json file"
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("") / "攻击鲁棒性分析\cifar-10\scaling0.5.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=DEFAULT_T0,
        help="T0 for direct shrinkage of expected contribution toward base_weight.",
    )
    parser.add_argument(
        "--history-prior-count",
        type=float,
        default=DEFAULT_HISTORY_PRIOR_COUNT,
        help="Pseudo-count m for Bayesian shrinkage of each client's historical average weight toward base_weight.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_json_path = derive_output_json_path(args.input_json, args.output_json)
    build_subjective_logic(args.input_json, output_json_path, args.t0, args.history_prior_count)
    print(f"Saved subjective logic results to: {Path(output_json_path).resolve()}")


if __name__ == "__main__":
    main()
