import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import FitRes, MetricsAggregationFn


SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "0.0.0.0:8080")
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
MIN_AVAILABLE_CLIENTS = int(os.getenv("MIN_AVAILABLE_CLIENTS", "2"))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "2"))
RUN_NAME = os.getenv("RUN_NAME", "default_run")
ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "artifacts"))
SERVER_RUN_DIR = ARTIFACT_ROOT / RUN_NAME / "server"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    ensure_directory(path.parent)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(num_examples * float(metric_dict["accuracy"]) for num_examples, metric_dict in metrics)
    return {"accuracy": weighted_accuracy / total_examples}


def fit_config(server_round: int) -> Dict[str, int]:
    return {
        "server_round": server_round,
        "local_epochs": LOCAL_EPOCHS,
    }


def evaluate_config(server_round: int) -> Dict[str, int]:
    return {"server_round": server_round}


class ServerArtifactLogger:
    def __init__(self) -> None:
        ensure_directory(SERVER_RUN_DIR)
        self.fit_csv_path = SERVER_RUN_DIR / "fit_rounds.csv"
        self.eval_csv_path = SERVER_RUN_DIR / "evaluation_rounds.csv"
        self.client_round_csv_path = SERVER_RUN_DIR / "client_rounds.csv"
        self.client_round_fieldnames = [
            "round",
            "phase",
            "hospital_name",
            "data_file",
            "num_examples",
            "train_loss",
            "train_accuracy",
            "tensor_count",
            "scalar_count",
            "loss",
            "accuracy",
        ]
        self.summary_json_path = SERVER_RUN_DIR / "summary.json"
        self.history: List[Dict[str, object]] = []
        self.metadata = {
            "run_name": RUN_NAME,
            "server_address": SERVER_ADDRESS,
            "num_rounds": NUM_ROUNDS,
            "min_available_clients": MIN_AVAILABLE_CLIENTS,
            "local_epochs": LOCAL_EPOCHS,
            "aggregation_strategy": "fedavg",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.flush()

    def record_fit_round(self, server_round: int, metrics: Dict[str, object], client_count: int) -> None:
        row = {
            "round": server_round,
            "client_count": client_count,
            **metrics,
        }
        self.history.append({"phase": "fit", **row})
        append_csv_row(self.fit_csv_path, list(row.keys()), row)
        self.flush()

    def record_eval_round(self, server_round: int, loss: float, metrics: Dict[str, object], client_count: int) -> None:
        row = {
            "round": server_round,
            "client_count": client_count,
            "loss": loss,
            **metrics,
        }
        self.history.append({"phase": "evaluate", **row})
        append_csv_row(self.eval_csv_path, list(row.keys()), row)
        self.flush()

    def record_client_round(self, server_round: int, phase: str, metrics: Dict[str, object], num_examples: int) -> None:
        row = {
            "round": server_round,
            "phase": phase,
            "hospital_name": metrics.get("hospital_name"),
            "data_file": metrics.get("data_file"),
            "num_examples": num_examples,
            **{
                key: value
                for key, value in metrics.items()
                if key not in {"hospital_name", "data_file"}
            },
        }
        normalized_row = {fieldname: row.get(fieldname, "") for fieldname in self.client_round_fieldnames}
        append_csv_row(self.client_round_csv_path, self.client_round_fieldnames, normalized_row)

    def flush(self) -> None:
        final_fit = next((item for item in reversed(self.history) if item["phase"] == "fit"), None)
        final_evaluate = next((item for item in reversed(self.history) if item["phase"] == "evaluate"), None)
        write_json(
            self.summary_json_path,
            {
                "metadata": self.metadata,
                "final_fit": final_fit,
                "final_evaluate": final_evaluate,
                "history": self.history,
            },
        )


SERVER_ARTIFACT_LOGGER = ServerArtifactLogger()


class LoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, metrics_aggregation_fn: MetricsAggregationFn, **kwargs) -> None:
        super().__init__(*args, evaluate_metrics_aggregation_fn=metrics_aggregation_fn, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"[SERVER] Round {server_round}: selecting clients for local training.")
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)
        print(f"[SERVER] Round {server_round}: sent global model to {len(fit_instructions)} client(s).")
        return fit_instructions

    def aggregate_fit(self, server_round, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], failures):
        print(f"[SERVER] Round {server_round}: received {len(results)} client update(s). Applying FedAvg.")
        for client_proxy, fit_res in results:
            metrics = dict(fit_res.metrics)
            SERVER_ARTIFACT_LOGGER.record_client_round(server_round, "fit", metrics, fit_res.num_examples)
            print(
                f"[SERVER] Round {server_round}: update from {metrics.get('hospital_name', client_proxy.cid)} "
                f"examples={fit_res.num_examples}, train_accuracy={float(metrics.get('train_accuracy', 0.0)):.4f}"
            )

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if results:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            weighted_train_accuracy = sum(
                fit_res.num_examples * float(fit_res.metrics.get("train_accuracy", 0.0))
                for _, fit_res in results
            ) / total_examples
            SERVER_ARTIFACT_LOGGER.record_fit_round(
                server_round,
                {
                    "train_accuracy": weighted_train_accuracy,
                    "received_client_updates": len(results),
                    "aggregation": "fedavg",
                },
                len(results),
            )
            print(f"[SERVER] Round {server_round}: FedAvg complete.")

        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        evaluate_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        print(f"[SERVER] Round {server_round}: sent global model for evaluation to {len(evaluate_instructions)} client(s).")
        return evaluate_instructions

    def aggregate_evaluate(self, server_round, results, failures):
        for _, evaluate_res in results:
            SERVER_ARTIFACT_LOGGER.record_client_round(
                server_round,
                "evaluate",
                {"loss": evaluate_res.loss, **dict(evaluate_res.metrics)},
                evaluate_res.num_examples,
            )
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        if aggregated_loss is not None:
            SERVER_ARTIFACT_LOGGER.record_eval_round(
                server_round,
                float(aggregated_loss),
                aggregated_metrics or {},
                len(results),
            )
            accuracy = float((aggregated_metrics or {}).get("accuracy", 0.0))
            print(f"[SERVER] Round {server_round}: global evaluation accuracy={accuracy:.4f}, loss={aggregated_loss:.4f}")
        return aggregated_loss, aggregated_metrics


def start_federated_server() -> None:
    print(f"[SERVER] Launching FedAvg server on {SERVER_ADDRESS}")
    print(f"[SERVER] Waiting for {MIN_AVAILABLE_CLIENTS} client(s). Rounds={NUM_ROUNDS}, local_epochs={LOCAL_EPOCHS}")
    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_AVAILABLE_CLIENTS,
        min_evaluate_clients=MIN_AVAILABLE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        metrics_aggregation_fn=weighted_average,
    )
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_federated_server()
