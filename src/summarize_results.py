import argparse
import csv
from collections import defaultdict
from pathlib import Path


def safe_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/baseline_results.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Result file not found: {input_path}")

    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "exact_match_sum": 0.0,
            "f1_sum": 0.0,
            "retrieval_calls_sum": 0.0,
            "doc_count_sum": 0.0,
        }
    )
    reason_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["baseline"]
            grouped[name]["count"] += 1
            grouped[name]["exact_match_sum"] += safe_float(row.get("exact_match", "0"))
            grouped[name]["f1_sum"] += safe_float(row.get("f1", "0"))
            grouped[name]["retrieval_calls_sum"] += safe_float(row.get("retrieval_calls", "0"))
            grouped[name]["doc_count_sum"] += safe_float(row.get("doc_count", "0"))
            reason = row.get("reason", "")
            if reason:
                reason_counts[name][reason] += 1

    print("=== Baseline Summary ===")
    print(f"{'baseline':30} {'n':>4} {'EM':>8} {'F1':>8} {'calls':>8} {'docs':>8}")
    for name, stats in grouped.items():
        count = max(stats["count"], 1.0)
        avg_em = stats["exact_match_sum"] / count
        avg_f1 = stats["f1_sum"] / count
        avg_calls = stats["retrieval_calls_sum"] / count
        avg_docs = stats["doc_count_sum"] / count
        print(
            f"{name:30} {int(count):4d} "
            f"{avg_em:8.3f} {avg_f1:8.3f} {avg_calls:8.3f} {avg_docs:8.3f}"
        )

    print("\n=== Reason Counts ===")
    for name, counts in reason_counts.items():
        joined = ", ".join(f"{reason}: {count}" for reason, count in sorted(counts.items()))
        print(f"{name}: {joined}")


if __name__ == "__main__":
    main()
