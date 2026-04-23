import argparse
import csv
from collections import defaultdict
from pathlib import Path


def safe_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def read_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_rows(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "exact_match_sum": 0.0,
            "f1_sum": 0.0,
            "retrieval_calls_sum": 0.0,
            "doc_count_sum": 0.0,
            "expanded_count": 0.0,
        }
    )
    reason_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        name = row["baseline"]
        grouped[name]["count"] += 1
        grouped[name]["exact_match_sum"] += safe_float(row.get("exact_match", "0"))
        grouped[name]["f1_sum"] += safe_float(row.get("f1", "0"))
        grouped[name]["retrieval_calls_sum"] += safe_float(row.get("retrieval_calls", "0"))
        grouped[name]["doc_count_sum"] += safe_float(row.get("doc_count", "0"))
        if row.get("decision") == "retrieve_more":
            grouped[name]["expanded_count"] += 1
        reason = row.get("reason", "")
        if reason:
            reason_counts[name][reason] += 1
    return grouped, reason_counts


def print_summary(grouped: dict[str, dict[str, float]], reason_counts: dict[str, dict[str, int]]) -> None:
    print("=== Baseline Summary ===")
    print(f"{'baseline':30} {'n':>4} {'EM':>8} {'F1':>8} {'calls':>8} {'docs':>8} {'expand%':>8}")
    for name, stats in sorted(grouped.items()):
        count = max(stats["count"], 1.0)
        avg_em = stats["exact_match_sum"] / count
        avg_f1 = stats["f1_sum"] / count
        avg_calls = stats["retrieval_calls_sum"] / count
        avg_docs = stats["doc_count_sum"] / count
        expand_rate = stats["expanded_count"] / count
        print(
            f"{name:30} {int(count):4d} "
            f"{avg_em:8.3f} {avg_f1:8.3f} {avg_calls:8.3f} {avg_docs:8.3f} {expand_rate:8.3f}"
        )

    print("\n=== Reason Counts ===")
    for name, counts in sorted(reason_counts.items()):
        joined = ", ".join(f"{reason}: {count}" for reason, count in sorted(counts.items()))
        print(f"{name}: {joined}")


def write_ablation_summary(rows: list[dict[str, str]], output: Path) -> None:
    grouped, _ = summarize_rows(rows)
    ablation_rows = []
    for baseline, stats in grouped.items():
        if baseline != "structure_aware_adaptive_rag" and not baseline.startswith("structure_aware_wo_"):
            continue
        count = max(stats["count"], 1.0)
        ablation_rows.append(
            {
                "baseline": baseline,
                "n": int(count),
                "em": stats["exact_match_sum"] / count,
                "f1": stats["f1_sum"] / count,
                "calls": stats["retrieval_calls_sum"] / count,
                "docs": stats["doc_count_sum"] / count,
                "expand_rate": stats["expanded_count"] / count,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["baseline", "n", "em", "f1", "calls", "docs", "expand_rate"]
        )
        writer.writeheader()
        for row in sorted(ablation_rows, key=lambda item: item["baseline"]):
            writer.writerow(row)
    print(f"Saved ablation summary to {output}")


def collect_inputs(single_input: str, inputs_pattern: str) -> list[Path]:
    if inputs_pattern:
        return sorted(Path(".").glob(inputs_pattern))
    return [Path(single_input)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/baseline_results.csv")
    parser.add_argument("--inputs-glob", default="")
    parser.add_argument("--ablation-output", default="")
    args = parser.parse_args()

    input_paths = collect_inputs(args.input, args.inputs_glob)
    if not input_paths:
        raise FileNotFoundError("No input files matched.")
    all_rows: list[dict[str, str]] = []
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Result file not found: {input_path}")
        all_rows.extend(read_rows(input_path))

    grouped, reason_counts = summarize_rows(all_rows)
    print_summary(grouped, reason_counts)

    if args.ablation_output:
        write_ablation_summary(all_rows, Path(args.ablation_output))


if __name__ == "__main__":
    main()
