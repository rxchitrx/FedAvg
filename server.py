import logging
import os
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import parameters_to_ndarrays


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("fedavg-server")


SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
MIN_AVAILABLE_CLIENTS = int(os.getenv("MIN_AVAILABLE_CLIENTS", "2"))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "2"))


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


def summarize_parameters(parameters) -> str:
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


class VerboseFedAvg(fl.server.strategy.FedAvg):
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

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if results:
            weighted_metrics = weighted_average(
                [(fit_res.num_examples, dict(fit_res.metrics)) for _, fit_res in results]
            )
            LOGGER.info(
                "[SERVER] Round %d | FedAvg aggregation complete with weighted client metrics: %s",
                server_round,
                weighted_metrics,
            )

        if aggregated_parameters is not None:
            LOGGER.info(
                "[SERVER] Round %d | aggregated global model ready: %s",
                server_round,
                summarize_parameters(aggregated_parameters),
            )

        LOGGER.info("[SERVER] Round %d | aggregated global model will be broadcast to clients", server_round)
        return aggregated_parameters, aggregated_metrics

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

        return aggregated_loss, aggregated_metrics


def start_federated_server() -> None:
    LOGGER.info("[SERVER] Launching verbose FedAvg aggregator on %s", SERVER_ADDRESS)
    LOGGER.info(
        "[SERVER] Runtime config | rounds=%d min_available_clients=%d local_epochs_per_client=%d",
        NUM_ROUNDS,
        MIN_AVAILABLE_CLIENTS,
        LOCAL_EPOCHS,
    )

    strategy = VerboseFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_AVAILABLE_CLIENTS,
        min_evaluate_clients=MIN_AVAILABLE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_federated_server()
