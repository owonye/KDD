import argparse
import csv
from collections import defaultdict
from pathlib import Path


def safe_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_row(rows: list[dict[str, str]], baseline: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("baseline") == baseline:
            return row
    return None


def summarize(input_path: Path) -> tuple[list[dict[str, str | int | float]], dict[str, int | float]]:
    rows = load_rows(input_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("query_id", "")].append(row)

    by_reason: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "cases": 0,
            "confidence_premature_stop": 0,
            "star_corrected_premature_stop": 0,
            "star_expands_confidence_stops": 0,
            "expanded_stop_improves_em": 0,
            "expanded_stop_improves_f1": 0,
            "expanded_stop_star_em_correct": 0,
            "expanded_stop_confidence_em_wrong": 0,
            "star_improves_em": 0,
            "star_improves_f1": 0,
            "star_hurts_em": 0,
            "star_hurts_f1": 0,
            "sum_em_delta": 0.0,
            "sum_f1_delta": 0.0,
        }
    )
    overall = {
        "queries": 0,
        "confidence_premature_stop": 0,
        "star_premature_stop": 0,
        "star_corrected_premature_stop": 0,
        "star_expands_confidence_stops": 0,
        "expanded_stop_improves_em": 0,
        "expanded_stop_improves_f1": 0,
        "star_improves_em": 0,
        "star_improves_f1": 0,
    }

    for group_rows in grouped.values():
        star = pick_row(group_rows, "structure_aware_adaptive_rag")
        confidence = pick_row(group_rows, "confidence_adaptive_rag")
        if not star or not confidence:
            continue

        overall["queries"] += 1
        reason = star.get("reason", "") or "unknown"
        bucket = by_reason[reason]
        bucket["cases"] += 1

        star_decision = star.get("decision", "")
        confidence_decision = confidence.get("decision", "")
        star_error = star.get("decision_error_type", "")
        confidence_error = confidence.get("decision_error_type", "")
        star_em = safe_float(star.get("exact_match"))
        confidence_em = safe_float(confidence.get("exact_match"))
        star_f1 = safe_float(star.get("f1"))
        confidence_f1 = safe_float(confidence.get("f1"))
        em_delta = star_em - confidence_em
        f1_delta = star_f1 - confidence_f1

        oracle_should_expand = star.get("oracle_should_expand", "")
        inferred_confidence_premature_stop = (
            confidence_error == "premature_stop"
            or (confidence_decision == "answer_now" and oracle_should_expand == "True")
        )
        inferred_star_correct = star_error == "correct" or star.get("decision_correct", "") == "1"

        if inferred_confidence_premature_stop:
            overall["confidence_premature_stop"] += 1
            bucket["confidence_premature_stop"] += 1
        if star_error == "premature_stop":
            overall["star_premature_stop"] += 1

        expands_vs_stop = star_decision == "retrieve_more" and confidence_decision == "answer_now"
        if expands_vs_stop:
            overall["star_expands_confidence_stops"] += 1
            bucket["star_expands_confidence_stops"] += 1
            if em_delta > 0:
                overall["expanded_stop_improves_em"] += 1
                bucket["expanded_stop_improves_em"] += 1
            if f1_delta > 0:
                overall["expanded_stop_improves_f1"] += 1
                bucket["expanded_stop_improves_f1"] += 1
            if star_em > 0:
                bucket["expanded_stop_star_em_correct"] += 1
            if confidence_em == 0:
                bucket["expanded_stop_confidence_em_wrong"] += 1

        corrected = expands_vs_stop and inferred_confidence_premature_stop and inferred_star_correct
        if corrected:
            overall["star_corrected_premature_stop"] += 1
            bucket["star_corrected_premature_stop"] += 1

        if em_delta > 0:
            overall["star_improves_em"] += 1
            bucket["star_improves_em"] += 1
        elif em_delta < 0:
            bucket["star_hurts_em"] += 1

        if f1_delta > 0:
            overall["star_improves_f1"] += 1
            bucket["star_improves_f1"] += 1
        elif f1_delta < 0:
            bucket["star_hurts_f1"] += 1

        bucket["sum_em_delta"] += em_delta
        bucket["sum_f1_delta"] += f1_delta

    summary_rows: list[dict[str, str | int | float]] = []
    for reason, stats in sorted(
        by_reason.items(),
        key=lambda item: (item[1]["star_corrected_premature_stop"], item[1]["star_improves_f1"]),
        reverse=True,
    ):
        cases = int(stats["cases"])
        summary_rows.append(
            {
                "reason": reason,
                "cases": cases,
                "confidence_premature_stop": int(stats["confidence_premature_stop"]),
                "star_corrected_premature_stop": int(stats["star_corrected_premature_stop"]),
                "star_expands_confidence_stops": int(stats["star_expands_confidence_stops"]),
                "expanded_stop_improves_em": int(stats["expanded_stop_improves_em"]),
                "expanded_stop_improves_f1": int(stats["expanded_stop_improves_f1"]),
                "expanded_stop_star_em_correct": int(stats["expanded_stop_star_em_correct"]),
                "expanded_stop_confidence_em_wrong": int(stats["expanded_stop_confidence_em_wrong"]),
                "star_improves_em": int(stats["star_improves_em"]),
                "star_improves_f1": int(stats["star_improves_f1"]),
                "star_hurts_em": int(stats["star_hurts_em"]),
                "star_hurts_f1": int(stats["star_hurts_f1"]),
                "avg_em_delta": float(stats["sum_em_delta"]) / max(cases, 1),
                "avg_f1_delta": float(stats["sum_f1_delta"]) / max(cases, 1),
            }
        )
    return summary_rows, overall


def write_summary(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reason",
        "cases",
        "confidence_premature_stop",
        "star_corrected_premature_stop",
        "star_expands_confidence_stops",
        "expanded_stop_improves_em",
        "expanded_stop_improves_f1",
        "expanded_stop_star_em_correct",
        "expanded_stop_confidence_em_wrong",
        "star_improves_em",
        "star_improves_f1",
        "star_hurts_em",
        "star_hurts_f1",
        "avg_em_delta",
        "avg_f1_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows, overall = summarize(input_path)

    print("=== Failure-Mode Summary ===")
    print(
        "queries={queries} confidence_premature_stop={confidence_premature_stop} "
        "star_premature_stop={star_premature_stop} "
        "star_corrected_premature_stop={star_corrected_premature_stop} "
        "star_expands_confidence_stops={star_expands_confidence_stops} "
        "expanded_stop_improves_f1={expanded_stop_improves_f1}".format(**overall)
    )
    print(
        f"{'reason':28s} {'cases':>6s} {'confEarly':>9s} {'corrected':>9s} "
        f"{'stop->exp':>9s} {'exp+F1':>7s} {'S>F1':>6s} {'S<F1':>6s} {'avgF1d':>8s}"
    )
    for row in rows:
        print(
            f"{str(row['reason']):28s} {int(row['cases']):6d} "
            f"{int(row['confidence_premature_stop']):9d} "
            f"{int(row['star_corrected_premature_stop']):9d} "
            f"{int(row['star_expands_confidence_stops']):9d} "
            f"{int(row['expanded_stop_improves_f1']):7d} "
            f"{int(row['star_improves_f1']):6d} "
            f"{int(row['star_hurts_f1']):6d} "
            f"{float(row['avg_f1_delta']):8.4f}"
        )

    if args.output:
        write_summary(Path(args.output), rows)
        print(f"Saved failure-mode summary to {args.output}")


if __name__ == "__main__":
    main()
