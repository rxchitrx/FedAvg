import csv
import json
import os
from pathlib import Path
from typing import Dict, List


ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "artifacts"))
OUTPUT_DIR = ARTIFACT_ROOT / "comparisons"
OUTPUT_JSON = OUTPUT_DIR / "run_comparison.json"
OUTPUT_CSV = OUTPUT_DIR / "run_comparison.csv"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_run_summaries() -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    if not ARTIFACT_ROOT.exists():
        return summaries

    for run_dir in sorted(path for path in ARTIFACT_ROOT.iterdir() if path.is_dir() and path.name != "comparisons"):
        summary_path = run_dir / "server" / "summary.json"
        if not summary_path.exists():
            continue

        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata", {})
        final_fit = payload.get("final_fit") or {}
        final_eval = payload.get("final_evaluate") or {}

        summaries.append(
            {
                "run_name": metadata.get("run_name", run_dir.name),
                "aggregation_strategy": metadata.get("aggregation_strategy"),
                "num_rounds": metadata.get("num_rounds"),
                "final_fit_round": final_fit.get("round"),
                "final_eval_round": final_eval.get("round"),
                "final_eval_loss": final_eval.get("loss"),
                "final_eval_accuracy": final_eval.get("accuracy"),
                "final_train_accuracy": final_fit.get("train_accuracy"),
                "received_client_updates": final_fit.get("received_client_updates"),
            }
        )

    return summaries


def write_outputs(rows: List[Dict[str, object]]) -> None:
    ensure_directory(OUTPUT_DIR)

    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No completed server summaries found under artifacts/.")
        return

    headers = [
        "run_name",
        "aggregation_strategy",
        "num_rounds",
        "final_eval_accuracy",
        "final_eval_loss",
        "received_client_updates",
    ]
    widths = {header: max(len(header), *(len(str(row.get(header, ""))) for row in rows)) for header in headers}
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def main() -> None:
    rows = collect_run_summaries()
    write_outputs(rows)
    print_table(rows)
    print(f"\nJSON comparison: {OUTPUT_JSON}")
    print(f"CSV comparison:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
