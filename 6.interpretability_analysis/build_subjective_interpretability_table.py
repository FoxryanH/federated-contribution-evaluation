import argparse
import csv
import html
import json
import random
from pathlib import Path


DEFAULT_INPUT_JSON = (
    Path("主实验")
    / "cifar10"
    / "dirichlet_iid_contribution_analysis_contribution_t0_bayes_copy.json"
)
PREFERRED_CATEGORY_ORDER = ["clean", "noise20", "noise40", "noise60", "iid_shared_distribution"]
DEFAULT_MIN_ROUND = 20
DEFAULT_MAX_ROUND = 70
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample one client per category and one round per sampled client from a "
            "subjective-contribution JSON, then export a paper-style interpretability case table."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-round", type=int, default=DEFAULT_MIN_ROUND)
    parser.add_argument("--max-round", type=int, default=DEFAULT_MAX_ROUND)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, default=None)
    return parser.parse_args()


def load_json(json_path):
    with Path(json_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def derive_output_paths(input_json, output_csv, output_md, output_html):
    input_json = Path(input_json)
    stem = input_json.stem
    parent = input_json.parent
    csv_path = output_csv or (parent / f"{stem}_interpretability_case_table.csv")
    md_path = output_md or (parent / f"{stem}_interpretability_case_table.md")
    html_path = output_html or (parent / f"{stem}_interpretability_case_table.html")
    return Path(csv_path), Path(md_path), Path(html_path)


def build_client_summaries(data):
    if isinstance(data.get("client_average_contributions"), list):
        return data["client_average_contributions"]

    records = data.get("records", [])
    stats = {}
    for record in records:
        client_id = int(record["client_id"])
        entry = stats.setdefault(
            client_id,
            {
                "client_id": client_id,
                "client_category": record.get("client_category", "unknown"),
                "selected_count": 0,
            },
        )
        entry["selected_count"] += 1

    return [stats[client_id] for client_id in sorted(stats.keys())]


def ordered_categories(client_summaries):
    encountered = []
    for item in client_summaries:
        category = item.get("client_category", "unknown")
        if category not in encountered:
            encountered.append(category)

    ordered = [category for category in PREFERRED_CATEGORY_ORDER if category in encountered]
    ordered.extend([category for category in encountered if category not in ordered])
    return ordered


def choose_record_for_client(rng, records, min_round, max_round):
    preferred_records = [
        record
        for record in records
        if min_round <= int(record["round_index"]) <= max_round
    ]
    candidate_records = preferred_records if preferred_records else records
    return rng.choice(candidate_records)


def format_float(value, digits=4, none_text="无"):
    if value is None:
        return none_text
    return f"{float(value):.{digits}f}"


def participation_text(index):
    return f"第{int(index)}次"


def evidence_values(record):
    return {
        "p1": float(record.get("p1_weight_adv", 0.0)),
        "p2": float(record.get("p2_loo_pos_adv", 0.0)),
        "n1": float(record.get("n1_weight_drop_bad", 0.0)),
        "n2": float(record.get("n2_loo_neg_bad", 0.0)),
    }


def rank_text(rank_index, total_count):
    return f"同轮参与者中主观逻辑值位列第{rank_index}"


def build_weight_evidence_text(p1, n1):
    if p1 <= EPS and n1 <= EPS:
        return "p1=0，n1=0：未出现明显权重侧证据"
    if p1 > EPS and n1 <= EPS:
        return f"p1={p1:.4f}，n1=0：出现提升权重正证据"
    if p1 <= EPS and n1 > EPS:
        return f"p1=0，n1={n1:.4f}：出现降低权重负证据"
    if p1 >= n1:
        return f"p1={p1:.4f}，n1={n1:.4f}：权重正负证据并存，但以正证据为主"
    return f"p1={p1:.4f}，n1={n1:.4f}：权重正负证据并存，但以负证据为主"


def build_loo_evidence_text(p2, n2):
    if p2 <= EPS and n2 <= EPS:
        return "p2=0，n2=0：边际贡献证据不明显"
    if p2 > EPS and n2 <= EPS:
        return f"p2={p2:.4f}，n2=0：边际贡献为正证据"
    if p2 <= EPS and n2 > EPS:
        return f"p2=0，n2={n2:.4f}：边际贡献为负证据"
    if p2 >= n2:
        return f"p2={p2:.4f}，n2={n2:.4f}：边际贡献正负证据并存，但以正证据为主"
    return f"p2={p2:.4f}，n2={n2:.4f}：边际贡献正负证据并存，但以负证据为主"


def build_explanation(
    record,
    selected_round_text,
    round_rank_index,
    round_total_count,
    historical_weight,
):
    values = evidence_values(record)
    p1 = values["p1"]
    p2 = values["p2"]
    n1 = values["n1"]
    n2 = values["n2"]
    contribution = float(record.get("normalized_contribution", 0.0))
    fl_round = int(record["round_index"])
    weight_text = build_weight_evidence_text(p1, n1)
    loo_text = build_loo_evidence_text(p2, n2)
    peer_text = rank_text(round_rank_index, round_total_count)
    history_text = format_float(historical_weight, none_text="无")
    return (
        f"{weight_text}；{loo_text}；"
        f"结合当前FL第{fl_round}轮、该客户端{selected_round_text}被选中，"
        f"历史平均权重为{history_text}，且{peer_text}，"
        f"当轮贡献被映射为{contribution:.4f}。"
    )


def sample_case_rows(data, seed, min_round, max_round):
    rng = random.Random(seed)
    client_summaries = build_client_summaries(data)
    records = data.get("records", [])

    records_by_client = {}
    records_by_round = {}
    for record in records:
        client_id = int(record["client_id"])
        records_by_client.setdefault(client_id, []).append(record)
        round_index = int(record["round_index"])
        records_by_round.setdefault(round_index, []).append(record)

    summaries_by_client = {int(item["client_id"]): item for item in client_summaries}
    categories = ordered_categories(client_summaries)

    rows = []
    for category in categories:
        category_clients = [
            int(item["client_id"])
            for item in client_summaries
            if item.get("client_category") == category and int(item["client_id"]) in records_by_client
        ]
        if not category_clients:
            continue

        sampled_client_id = rng.choice(sorted(category_clients))
        sampled_record = choose_record_for_client(
            rng,
            records_by_client[sampled_client_id],
            min_round,
            max_round,
        )

        sampled_round = int(sampled_record["round_index"])
        client_rounds = sorted(int(item["round_index"]) for item in records_by_client[sampled_client_id])
        current_participation_index = client_rounds.index(sampled_round) + 1
        prior_participation_count = current_participation_index - 1
        client_summary = summaries_by_client[sampled_client_id]

        historical_avg_weight = None
        if prior_participation_count > 0:
            historical_avg_weight = sampled_record.get("historical_avg_weight")

        values = evidence_values(sampled_record)
        same_round_records = records_by_round.get(sampled_round, [])
        sorted_same_round_records = sorted(
            same_round_records,
            key=lambda item: float(
                item.get(
                    "expected_contribution",
                    item.get(
                        "expected_contribution_raw",
                        item.get("normalized_contribution", 0.0),
                    ),
                )
            ),
            reverse=True,
        )
        round_rank_index = 1
        for idx, item in enumerate(sorted_same_round_records, start=1):
            if int(item["client_id"]) == sampled_client_id:
                round_rank_index = idx
                break
        round_total_count = max(len(sorted_same_round_records), 1)
        selected_round_text = participation_text(current_participation_index)
        rows.append(
            {
                "id": sampled_client_id,
                "type": category,
                "fl_round": sampled_round,
                "selected_round": selected_round_text,
                "historical_weight": None if historical_avg_weight is None else float(historical_avg_weight),
                "optimized_weight": float(sampled_record.get("optimized_weight", 0.0)),
                "p1": values["p1"],
                "p2": values["p2"],
                "n1": values["n1"],
                "n2": values["n2"],
                "round_contribution": float(sampled_record.get("normalized_contribution", 0.0)),
                "explanation": build_explanation(
                    sampled_record,
                    selected_round_text,
                    round_rank_index,
                    round_total_count,
                    historical_avg_weight,
                ),
                "selected_count": int(client_summary.get("selected_count", len(client_rounds))),
            }
        )

    return rows


def build_markdown(rows, input_json, seed, min_round, max_round):
    lines = []
    lines.append("# 可解释性案例表")
    lines.append("")
    lines.append(f"- 数据来源：`{input_json}`")
    lines.append(f"- 随机种子：`{seed}`")
    lines.append("")
    lines.append(
        "| id | 类型 | FL轮次 | 被选中轮次 | 历史权重 | 优化后权重 | p1 | p2 | n1 | n2 | 当轮贡献值 | 解释 |"
    )
    lines.append("| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            "| {id} | {type} | {fl_round} | {selected_round} | {historical_weight} | {optimized_weight} | {p1} | {p2} | {n1} | {n2} | {round_contribution} | {explanation} |".format(
                id=row["id"],
                type=row["type"],
                fl_round=row["fl_round"],
                selected_round=row["selected_round"],
                historical_weight=format_float(row["historical_weight"], none_text="无"),
                optimized_weight=format_float(row["optimized_weight"]),
                p1=format_float(row["p1"]),
                p2=format_float(row["p2"]),
                n1=format_float(row["n1"]),
                n2=format_float(row["n2"]),
                round_contribution=format_float(row["round_contribution"]),
                explanation=row["explanation"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_html(rows, input_json, seed, min_round, max_round):
    row_html = []
    for row in rows:
        row_html.append(
            "<tr>"
            f"<td>{row['id']}</td>"
            f"<td>{html.escape(str(row['type']))}</td>"
            f"<td>{row['fl_round']}</td>"
            f"<td>{html.escape(str(row['selected_round']))}</td>"
            f"<td>{html.escape(format_float(row['historical_weight'], none_text='无'))}</td>"
            f"<td>{html.escape(format_float(row['optimized_weight']))}</td>"
            f"<td>{html.escape(format_float(row['p1']))}</td>"
            f"<td>{html.escape(format_float(row['p2']))}</td>"
            f"<td>{html.escape(format_float(row['n1']))}</td>"
            f"<td>{html.escape(format_float(row['n2']))}</td>"
            f"<td>{html.escape(format_float(row['round_contribution']))}</td>"
            f"<td class='explain'>{html.escape(row['explanation'])}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>可解释性案例表</title>
  <style>
    body {{
      margin: 24px;
      font-family: "Times New Roman", "SimSun", serif;
      color: #111;
      background: #fff;
    }}
    .meta {{
      margin-bottom: 14px;
      font-size: 14px;
      line-height: 1.5;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      table-layout: fixed;
      border: 1.5px solid #000;
    }}
    th, td {{
      border: 1px solid #000;
      padding: 8px 6px;
      text-align: center;
      vertical-align: middle;
      font-size: 14px;
      line-height: 1.5;
      word-break: break-word;
    }}
    thead th {{
      font-weight: 600;
    }}
    .explain {{
      text-align: left;
      padding-left: 10px;
      padding-right: 10px;
    }}
    .narrow {{ width: 5.5%; }}
    .mid {{ width: 7.5%; }}
    .wide {{ width: 9.5%; }}
    .explain-col {{ width: 27%; }}
  </style>
</head>
<body>
  <div class="meta">
    <div><strong>数据来源：</strong>{html.escape(str(input_json))}</div>
    <div><strong>随机种子：</strong>{seed}</div>
  </div>

  <table>
    <thead>
      <tr>
        <th rowspan="2" class="narrow">id</th>
        <th rowspan="2" class="narrow">类型</th>
        <th rowspan="2" class="wide">FL<br>轮次</th>
        <th rowspan="2" class="wide">被选中<br>轮次</th>
        <th rowspan="2" class="wide">历史<br>权重</th>
        <th rowspan="2" class="wide">优化后<br>权重</th>
        <th colspan="2" class="mid">正证据</th>
        <th colspan="2" class="mid">负证据</th>
        <th rowspan="2" class="wide">当轮贡<br>献值</th>
        <th rowspan="2" class="explain-col">解释</th>
      </tr>
      <tr>
        <th class="mid">p1</th>
        <th class="mid">p2</th>
        <th class="mid">n1</th>
        <th class="mid">n2</th>
      </tr>
    </thead>
    <tbody>
      {''.join(row_html)}
    </tbody>
  </table>
</body>
</html>
"""


def write_csv(rows, csv_path):
    fieldnames = [
        "id",
        "type",
        "fl_round",
        "selected_round",
        "historical_weight",
        "optimized_weight",
        "p1",
        "p2",
        "n1",
        "n2",
        "round_contribution",
        "explanation",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main():
    args = parse_args()
    data = load_json(args.input_json)
    rows = sample_case_rows(data, args.seed, args.min_round, args.max_round)
    if not rows:
        raise RuntimeError("No interpretable rows could be sampled from the input JSON.")

    output_csv, output_md, output_html = derive_output_paths(
        args.input_json,
        args.output_csv,
        args.output_md,
        args.output_html,
    )

    markdown = build_markdown(rows, args.input_json, args.seed, args.min_round, args.max_round)
    html_text = build_html(rows, args.input_json, args.seed, args.min_round, args.max_round)

    output_md.write_text(markdown, encoding="utf-8")
    output_html.write_text(html_text, encoding="utf-8")
    write_csv(rows, output_csv)

    print(f"Case-table markdown saved to: {output_md}")
    print(f"Case-table html saved to: {output_html}")
    print(f"Case-table csv saved to: {output_csv}")


if __name__ == "__main__":
    main()
