import argparse
import json
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path("artifacts")


def format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def latest_audit_path(run_name: str) -> Path:
    audit_dir = ARTIFACT_ROOT / run_name / "server" / "update_audits"
    paths = sorted(audit_dir.glob("round_*_update_audit.json"))
    if not paths:
        raise FileNotFoundError(f"No update audit JSON files found under {audit_dir}")
    return paths[-1]


def print_audit(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"Update audit: {path}")
    print(f"Run: {payload.get('run_name')} | Round: {payload.get('round')} | Strategy: {payload.get('aggregation_strategy')}")
    print(
        "Detection: "
        f"{payload.get('detection_enabled')} | "
        f"accepted={payload.get('accepted_client_count')} | "
        f"rejected={payload.get('rejected_client_count')}"
    )
    print()

    headers = ["hospital", "behavior", "decision", "tensors", "scalars", "mean", "std", "l2_norm", "preview"]
    rows = []
    for client in payload.get("clients", []):
        summary = client.get("parameter_summary", {})
        overall = summary.get("overall", {})
        preview = ", ".join(format_number(value, 4) for value in overall.get("preview_values", [])[:5])
        rows.append(
            [
                client.get("hospital_name") or client.get("client_id"),
                client.get("client_behavior"),
                client.get("aggregation_decision"),
                summary.get("tensor_count"),
                summary.get("scalar_count"),
                format_number(overall.get("mean")),
                format_number(overall.get("std")),
                format_number(overall.get("l2_norm")),
                preview,
            ]
        )

    widths = {
        header: max(len(header), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[headers[index]]) for index, value in enumerate(row)))

    aggregated = payload.get("aggregated_parameter_summary")
    if aggregated:
        overall = aggregated.get("overall", {})
        print()
        print(
            "Aggregated global update: "
            f"tensors={aggregated.get('tensor_count')} | "
            f"scalars={aggregated.get('scalar_count')} | "
            f"mean={format_number(overall.get('mean'))} | "
            f"std={format_number(overall.get('std'))} | "
            f"l2_norm={format_number(overall.get('l2_norm'))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a readable summary of saved client update audit artifacts.")
    parser.add_argument("run_name", help="Artifact run name, for example update_audit_demo")
    parser.add_argument("--round", type=int, help="Round number to inspect. Defaults to the latest audit round.")
    args = parser.parse_args()

    if args.round is None:
        path = latest_audit_path(args.run_name)
    else:
        path = ARTIFACT_ROOT / args.run_name / "server" / "update_audits" / f"round_{args.round:03d}_update_audit.json"
        if not path.exists():
            raise FileNotFoundError(f"No audit file found at {path}")

    print_audit(path)


if __name__ == "__main__":
    main()
