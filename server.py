import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("fedavg-server")


SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
MIN_AVAILABLE_CLIENTS = int(os.getenv("MIN_AVAILABLE_CLIENTS", "2"))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "2"))
AGGREGATION_STRATEGY = os.getenv("AGGREGATION_STRATEGY", "fedavg").strip().lower()
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

    totals: Dict[str, float] = {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    for num_examples, metric_dict in metrics:
        for metric_name, metric_value in metric_dict.items():
            totals[metric_name] = totals.get(metric_name, 0.0) + (num_examples * float(metric_value))

    return {
        metric_name: metric_total / total_examples
        for metric_name, metric_total in totals.items()
    }


def summarize_parameters(parameters: Parameters) -> str:
    ndarrays = parameters_to_ndarrays(parameters)
    total_scalars = sum(int(array.size) for array in ndarrays)
    return f"{len(ndarrays)} tensors | {total_scalars:,} scalar values"


def fit_config(server_round: int) -> Dict[str, int]:
    return {
        "server_round": server_round,
        "local_epochs": LOCAL_EPOCHS,
    }


def evaluate_config(server_round: int) -> Dict[str, int]:
    return {
        "server_round": server_round,
    }


def coordinate_wise_median(results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]]) -> Parameters:
    client_ndarrays: List[NDArrays] = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
    aggregated: NDArrays = []
    for layers in zip(*client_ndarrays):
        aggregated.append(np.median(np.stack(layers, axis=0), axis=0))
    return ndarrays_to_parameters(aggregated)


class ServerArtifactLogger:
    def __init__(self) -> None:
        self.started_at = datetime.utcnow().isoformat() + "Z"
        self.metadata = {
            "run_name": RUN_NAME,
            "server_address": SERVER_ADDRESS,
            "num_rounds": NUM_ROUNDS,
            "min_available_clients": MIN_AVAILABLE_CLIENTS,
            "local_epochs": LOCAL_EPOCHS,
            "aggregation_strategy": AGGREGATION_STRATEGY,
            "started_at": self.started_at,
        }
        self.round_records: List[Dict[str, object]] = []
        self.fit_csv_path = SERVER_RUN_DIR / "fit_rounds.csv"
        self.eval_csv_path = SERVER_RUN_DIR / "evaluation_rounds.csv"
        self.summary_json_path = SERVER_RUN_DIR / "summary.json"
        ensure_directory(SERVER_RUN_DIR)

    def record_fit_round(self, round_number: int, metrics: Dict[str, float], client_count: int) -> None:
        row = {
            "round": round_number,
            "client_count": client_count,
            **metrics,
        }
        self.round_records.append({"phase": "fit", **row})
        append_csv_row(self.fit_csv_path, list(row.keys()), row)
        self.flush()

    def record_eval_round(
        self,
        round_number: int,
        loss: float,
        metrics: Dict[str, float],
        client_count: int,
    ) -> None:
        row = {
            "round": round_number,
            "client_count": client_count,
            "loss": loss,
            **metrics,
        }
        self.round_records.append({"phase": "evaluate", **row})
        append_csv_row(self.eval_csv_path, list(row.keys()), row)
        self.flush()

    def flush(self) -> None:
        final_fit = next((item for item in reversed(self.round_records) if item["phase"] == "fit"), None)
        final_eval = next((item for item in reversed(self.round_records) if item["phase"] == "evaluate"), None)
        write_json(
            self.summary_json_path,
            {
                "metadata": self.metadata,
                "final_fit": final_fit,
                "final_evaluate": final_eval,
                "history": self.round_records,
            },
        )


SERVER_ARTIFACT_LOGGER = ServerArtifactLogger()


class VerboseFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, metrics_aggregation_fn: MetricsAggregationFn, **kwargs) -> None:
        super().__init__(*args, evaluate_metrics_aggregation_fn=metrics_aggregation_fn, **kwargs)
        self.metrics_aggregation_fn = metrics_aggregation_fn

    def configure_fit(self, server_round, parameters, client_manager):
        LOGGER.info("[SERVER] Round %d | preparing fit instructions for clients", server_round)
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        LOGGER.info("[SERVER] Round %d | selected %d client(s) for training", server_round, len(fit_ins))
        for client_proxy, _ in fit_ins:
            LOGGER.info("[SERVER] Round %d | dispatched training request to client %s", server_round, client_proxy.cid)
        return fit_ins

    def aggregate_fit(self, server_round, results, failures):
        LOGGER.info("[SERVER] Round %d | waiting for local model updates completed", server_round)
        LOGGER.info(
            "[SERVER] Round %d | received %d fit result(s), failures=%d",
            server_round,
            len(results),
            len(failures),
        )
        for client_proxy, fit_res in results:
            LOGGER.info(
                "[SERVER] Round %d | client %s update received: examples=%d metrics=%s",
                server_round,
                client_proxy.cid,
                fit_res.num_examples,
                dict(fit_res.metrics),
            )

        aggregated_parameters, aggregated_metrics = self._aggregate_parameters(server_round, results, failures)
        weighted_metrics = {}
        if results:
            weighted_metrics = self.metrics_aggregation_fn(
                [(fit_res.num_examples, dict(fit_res.metrics)) for _, fit_res in results]
            )
            LOGGER.info(
                "[SERVER] Round %d | %s aggregation complete with weighted client metrics: %s",
                server_round,
                AGGREGATION_STRATEGY.upper(),
                weighted_metrics,
            )
            SERVER_ARTIFACT_LOGGER.record_fit_round(server_round, weighted_metrics, len(results))

        if aggregated_parameters is not None:
            LOGGER.info(
                "[SERVER] Round %d | aggregated global model ready: %s",
                server_round,
                summarize_parameters(aggregated_parameters),
            )

        LOGGER.info("[SERVER] Round %d | aggregated global model will be broadcast to clients", server_round)
        return aggregated_parameters, aggregated_metrics

    def _aggregate_parameters(self, server_round, results, failures):
        return super().aggregate_fit(server_round, results, failures)

    def configure_evaluate(self, server_round, parameters, client_manager):
        LOGGER.info("[SERVER] Round %d | preparing federated evaluation request", server_round)
        evaluate_ins = super().configure_evaluate(server_round, parameters, client_manager)
        LOGGER.info("[SERVER] Round %d | selected %d client(s) for evaluation", server_round, len(evaluate_ins))
        for client_proxy, _ in evaluate_ins:
            LOGGER.info("[SERVER] Round %d | dispatched evaluation request to client %s", server_round, client_proxy.cid)
        return evaluate_ins

    def aggregate_evaluate(self, server_round, results, failures):
        LOGGER.info(
            "[SERVER] Round %d | received %d evaluation result(s), failures=%d",
            server_round,
            len(results),
            len(failures),
        )
        for client_proxy, evaluate_res in results:
            LOGGER.info(
                "[SERVER] Round %d | client %s evaluation received: loss=%.4f examples=%d metrics=%s",
                server_round,
                client_proxy.cid,
                evaluate_res.loss,
                evaluate_res.num_examples,
                dict(evaluate_res.metrics),
            )

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        if aggregated_loss is not None:
            LOGGER.info(
                "[SERVER] Round %d | aggregated federated evaluation: loss=%.4f metrics=%s",
                server_round,
                aggregated_loss,
                aggregated_metrics,
            )
            SERVER_ARTIFACT_LOGGER.record_eval_round(
                server_round,
                aggregated_loss,
                aggregated_metrics or {},
                len(results),
            )
        return aggregated_loss, aggregated_metrics


class VerboseMedianStrategy(VerboseFedAvg):
    def _aggregate_parameters(self, server_round, results, failures):
        if failures:
            LOGGER.warning(
                "[SERVER] Round %d | proceeding with median aggregation despite failures=%d",
                server_round,
                len(failures),
            )
        if not results:
            LOGGER.warning("[SERVER] Round %d | no client updates available for median aggregation", server_round)
            return None, {}

        LOGGER.info("[SERVER] Round %d | computing coordinate-wise median across client updates", server_round)
        aggregated_parameters = coordinate_wise_median(results)
        return aggregated_parameters, {}


def build_strategy() -> VerboseFedAvg:
    if AGGREGATION_STRATEGY not in {"fedavg", "median"}:
        raise ValueError("AGGREGATION_STRATEGY must be either 'fedavg' or 'median'")

    strategy_cls = VerboseFedAvg if AGGREGATION_STRATEGY == "fedavg" else VerboseMedianStrategy
    return strategy_cls(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_AVAILABLE_CLIENTS,
        min_evaluate_clients=MIN_AVAILABLE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        metrics_aggregation_fn=weighted_average,
    )


def start_federated_server() -> None:
    LOGGER.info("[SERVER] Launching verbose %s aggregator on %s", AGGREGATION_STRATEGY.upper(), SERVER_ADDRESS)
    LOGGER.info(
        "[SERVER] Runtime config | rounds=%d min_available_clients=%d local_epochs_per_client=%d run_name=%s",
        NUM_ROUNDS,
        MIN_AVAILABLE_CLIENTS,
        LOCAL_EPOCHS,
        RUN_NAME,
    )

    strategy = build_strategy()
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_federated_server()
