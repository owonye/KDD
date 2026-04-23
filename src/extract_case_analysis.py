import argparse
import csv
from pathlib import Path


def safe_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_structure_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        baseline = row.get("baseline", "")
        if baseline == "structure_aware_adaptive_rag" or baseline.startswith("structure_aware_wo_"):
            return row
    return None


def pick_confidence_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row.get("baseline") == "confidence_adaptive_rag":
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/case_analysis.csv")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        query_id = row.get("query_id", "")
        grouped.setdefault(query_id, []).append(row)

    candidates: list[dict[str, str | float]] = []
    for query_id, group_rows in grouped.items():
        structure = pick_structure_row(group_rows)
        confidence = pick_confidence_row(group_rows)
        if not structure or not confidence:
            continue

        decision_disagree = structure.get("decision") != confidence.get("decision")
        reason = structure.get("reason", "")
        structure_score = safe_float(structure.get("sufficiency_score"))
        confidence_score = safe_float(confidence.get("sufficiency_score"))
        score_gap = abs(structure_score - confidence_score)
        if not decision_disagree and reason not in {"low_coverage", "high_redundancy", "weak_supportiveness"}:
            continue

        candidates.append(
            {
                "query_id": query_id,
                "query": structure.get("query", ""),
                "gold_answer": structure.get("gold_answer", ""),
                "structure_baseline": structure.get("baseline", ""),
                "structure_decision": structure.get("decision", ""),
                "structure_reason": reason,
                "structure_score": structure_score,
                "confidence_decision": confidence.get("decision", ""),
                "confidence_score": confidence_score,
                "score_gap": score_gap,
                "decision_disagree": int(decision_disagree),
            }
        )

    ranked = sorted(candidates, key=lambda row: (row["decision_disagree"], row["score_gap"]), reverse=True)[: args.top_n]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query",
                "gold_answer",
                "structure_baseline",
                "structure_decision",
                "structure_reason",
                "structure_score",
                "confidence_decision",
                "confidence_score",
                "score_gap",
                "decision_disagree",
            ],
        )
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)
    print(f"Saved case analysis to {output_path}")


if __name__ == "__main__":
    main()
