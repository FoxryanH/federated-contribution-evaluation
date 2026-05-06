import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_CLIENT_COUNTS = [3, 4, 5, 6, 7]
DEFAULT_ROUNDS = 10
DEFAULT_CLIENT_SIZE = 1000


def parse_client_counts(raw_value):
    tokens = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    if not tokens:
        raise ValueError("client-counts cannot be empty.")
    client_counts = []
    for token in tokens:
        value = int(token)
        if value <= 0:
            raise ValueError("each client count must be positive.")
        client_counts.append(value)
    return client_counts


def build_detail_output_path(method_tag, num_clients):
    return Path("outputs") / "time_analysis" / method_tag / f"s1_{num_clients}clients_results.json"


def inject_timing_into_detail_json(
    detail_output_path,
    *,
    method_tag,
    code_version,
    num_clients,
    rounds,
    client_size,
    elapsed_seconds,
):
    if not detail_output_path.exists():
        return

    with detail_output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    payload["wall_clock_time_analysis"] = {
        "method": method_tag,
        "time_analysis_code_version": code_version,
        "n": int(num_clients),
        "rounds": int(rounds),
        "client_size": int(client_size),
        "wall_clock_seconds": float(elapsed_seconds),
        "wall_clock_minutes": float(elapsed_seconds / 60.0),
    }

    config = payload.get("config")
    if isinstance(config, dict):
        config["wall_clock_seconds"] = float(elapsed_seconds)
        config["wall_clock_minutes"] = float(elapsed_seconds / 60.0)

    with detail_output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def strip_detail_json_to_time_only(
    detail_output_path,
    *,
    method_tag,
    code_version,
    num_clients,
    rounds,
    client_size,
):
    if not detail_output_path.exists():
        return

    with detail_output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    minimal_payload = {
        "method": method_tag,
        "code_version": code_version,
        "n": int(num_clients),
        "rounds": int(rounds),
        "client_size": int(client_size),
        "parallel_time_analysis": payload.get("parallel_time_analysis", {}),
        "wall_clock_time_analysis": payload.get("wall_clock_time_analysis", {}),
    }

    with detail_output_path.open("w", encoding="utf-8") as handle:
        json.dump(minimal_payload, handle, ensure_ascii=False, indent=2)


def load_parallel_time_seconds(detail_output_path):
    if not detail_output_path.exists():
        return None
    with detail_output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    parallel_time_analysis = payload.get("parallel_time_analysis", {})
    total_parallel_seconds = parallel_time_analysis.get("total_estimated_parallel_time_seconds")
    if total_parallel_seconds is None:
        return None
    return float(total_parallel_seconds)


def resolve_base_script_path(base_script_name):
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates = [
        script_dir / base_script_name,
        repo_root / base_script_name,
        repo_root / "1.accuracy_analysis" / "controlled_experiments" / base_script_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Base script not found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def run_time_analysis(
    *,
    base_script_name,
    method_tag,
    code_version,
    default_output_json,
    description,
):
    parser = argparse.ArgumentParser(
        description=description,
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-json", type=Path, default=default_output_json)
    parser.add_argument("--scenario-tag", type=str, default="S1")
    parser.add_argument(
        "--client-counts",
        type=str,
        default="3,4,5,6,7",
        help="Comma-separated participant counts to benchmark.",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--client-size", type=int, default=DEFAULT_CLIENT_SIZE)
    parser.add_argument("--client-noise-rates", type=str, default="0")
    parser.add_argument("--client-missing-classes", type=str, default="0")
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args, passthrough_args = parser.parse_known_args()

    if args.rounds <= 0:
        raise ValueError("rounds must be positive.")
    if args.client_size <= 0:
        raise ValueError("client-size must be positive.")

    client_counts = parse_client_counts(args.client_counts)
    base_script_path = resolve_base_script_path(base_script_name)

    output_path = args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Controlled time-analysis setup:",
        f"code_version={code_version}",
        f"method={method_tag}",
        f"base_script={base_script_name}",
        f"scenario={args.scenario_tag}",
        f"client_counts={client_counts}",
        f"rounds={args.rounds}",
        f"client_size={args.client_size}",
    )

    start_wall_time = time.perf_counter()
    run_records = []

    for index, num_clients in enumerate(client_counts, start=1):
        detail_output_path = build_detail_output_path(method_tag, num_clients)
        detail_output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(base_script_path),
            "--data-dir",
            str(args.data_dir),
            "--scenario-tag",
            args.scenario_tag,
            "--num-clients",
            str(num_clients),
            "--clients-per-round",
            str(num_clients),
            "--rounds",
            str(args.rounds),
            "--client-data-sizes",
            str(args.client_size),
            "--client-noise-rates",
            args.client_noise_rates,
            "--client-missing-classes",
            args.client_missing_classes,
            "--output-json",
            str(detail_output_path),
        ]

        if args.gpu_ids is not None:
            command.extend(["--gpu-ids", args.gpu_ids])
        if args.num_workers is not None:
            command.extend(["--num-workers", str(args.num_workers)])
        command.extend(passthrough_args)

        print(
            f"[{method_tag} {index:02d}/{len(client_counts):02d}] start:",
            f"num_clients={num_clients}",
            f"rounds={args.rounds}",
            f"client_size={args.client_size}",
        )
        run_start = time.perf_counter()
        subprocess.run(command, check=True)
        elapsed_seconds = time.perf_counter() - run_start
        inject_timing_into_detail_json(
            detail_output_path,
            method_tag=method_tag,
            code_version=code_version,
            num_clients=num_clients,
            rounds=args.rounds,
            client_size=args.client_size,
            elapsed_seconds=elapsed_seconds,
        )
        strip_detail_json_to_time_only(
            detail_output_path,
            method_tag=method_tag,
            code_version=code_version,
            num_clients=num_clients,
            rounds=args.rounds,
            client_size=args.client_size,
        )
        parallel_time_seconds = load_parallel_time_seconds(detail_output_path)
        if parallel_time_seconds is None:
            parallel_time_seconds = float(elapsed_seconds)

        run_record = {
            "n": int(num_clients),
            "time_seconds": float(parallel_time_seconds),
            "time_minutes": float(parallel_time_seconds / 60.0),
        }
        run_records.append(run_record)

        print(
            f"[{method_tag} {index:02d}/{len(client_counts):02d}] finished:",
            f"num_clients={num_clients}",
            f"elapsed_seconds={elapsed_seconds:.4f}",
        )

    total_elapsed_seconds = time.perf_counter() - start_wall_time
    payload = {
        "config": {
            "code_version": code_version,
            "method": method_tag,
            "client_counts": client_counts,
            "rounds": args.rounds,
            "client_size": args.client_size,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "timing_results": run_records,
        "total_elapsed_seconds": float(total_elapsed_seconds),
        "total_elapsed_minutes": float(total_elapsed_seconds / 60.0),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Controlled time-analysis JSON saved to: {output_path}")
