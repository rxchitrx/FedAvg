import csv
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


HOSPITAL_NAME = os.getenv("HOSPITAL_NAME", "Hospital_A")
DATA_FILE = os.getenv("DATA_FILE", "hospital_A_data.npz")
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))
MOMENTUM = float(os.getenv("MOMENTUM", "0.9"))
RUN_NAME = os.getenv("RUN_NAME", "default_run")
ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "artifacts"))
CLIENT_RUN_DIR = ARTIFACT_ROOT / RUN_NAME / "clients" / HOSPITAL_NAME


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


class PneumoniaCNN(nn.Module):
    """Small CNN for binary pneumonia classification on 28x28 grayscale images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def accuracy_from_logits(outputs: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    _, predicted = torch.max(outputs.data, 1)
    correct = int((predicted == labels).sum().item())
    total = int(labels.size(0))
    return correct, total


def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer, criterion) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_correct, batch_total = accuracy_from_logits(outputs, labels)
        correct += batch_correct
        total += batch_total
        total_loss += float(loss.item()) * batch_total

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def evaluate(model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_correct, batch_total = accuracy_from_logits(outputs, labels)
            correct += batch_correct
            total += batch_total
            total_loss += float(loss.item()) * batch_total

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def summarize_parameters(parameters: List[np.ndarray]) -> str:
    total_scalars = sum(int(array.size) for array in parameters)
    return f"{len(parameters)} tensors | {total_scalars:,} scalar values"


class ClientArtifactLogger:
    def __init__(self) -> None:
        ensure_directory(CLIENT_RUN_DIR)
        self.round_history: List[Dict[str, object]] = []
        self.epoch_csv_path = CLIENT_RUN_DIR / "epoch_metrics.csv"
        self.round_json_path = CLIENT_RUN_DIR / "round_metrics.json"
        self.summary_json_path = CLIENT_RUN_DIR / "summary.json"
        self.metadata = {
            "hospital_name": HOSPITAL_NAME,
            "data_file": DATA_FILE,
            "server_address": SERVER_ADDRESS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.flush()

    def record_epoch(self, round_number: int, epoch_number: int, metrics: Dict[str, float]) -> None:
        row = {
            "round": round_number,
            "epoch": epoch_number,
            "train_loss": metrics["loss"],
            "train_accuracy": metrics["accuracy"],
        }
        append_csv_row(self.epoch_csv_path, list(row.keys()), row)

    def record_fit(self, round_number: int, metrics: Dict[str, float]) -> None:
        self.round_history.append({"round": round_number, "phase": "fit", **metrics})
        self.flush()

    def record_evaluate(self, round_number: int, metrics: Dict[str, float]) -> None:
        self.round_history.append({"round": round_number, "phase": "evaluate", **metrics})
        self.flush()

    def flush(self) -> None:
        write_json(
            self.round_json_path,
            {
                "metadata": self.metadata,
                "round_history": self.round_history,
            },
        )
        write_json(
            self.summary_json_path,
            {
                "metadata": self.metadata,
                "latest": self.round_history[-1] if self.round_history else {},
            },
        )


class PneumoniaClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, logger: ClientArtifactLogger) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger

    def get_parameters(self, config):
        return [value.detach().cpu().numpy() for _, value in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        round_number = int(config.get("server_round", 0))
        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        print(f"[{HOSPITAL_NAME}] Round {round_number}: global weights received. Training locally...")
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        final_train_metrics = {"loss": 0.0, "accuracy": 0.0}
        for epoch in range(1, local_epochs + 1):
            final_train_metrics = train_one_epoch(self.model, self.train_loader, optimizer, criterion)
            self.logger.record_epoch(round_number, epoch, final_train_metrics)
            print(
                f"[{HOSPITAL_NAME}] Round {round_number} Epoch {epoch}/{local_epochs}: "
                f"loss={final_train_metrics['loss']:.4f}, accuracy={final_train_metrics['accuracy']:.4f}"
            )

        outgoing_parameters = self.get_parameters(config={})
        print(f"[{HOSPITAL_NAME}] Sending update to server: {summarize_parameters(outgoing_parameters)}")
        fit_metrics = {
            "train_loss": final_train_metrics["loss"],
            "train_accuracy": final_train_metrics["accuracy"],
            "hospital_name": HOSPITAL_NAME,
            "data_file": DATA_FILE,
            "tensor_count": float(len(outgoing_parameters)),
            "scalar_count": float(sum(int(array.size) for array in outgoing_parameters)),
        }
        self.logger.record_fit(round_number, fit_metrics)
        return outgoing_parameters, len(self.train_loader.dataset), fit_metrics

    def evaluate(self, parameters, config):
        round_number = int(config.get("server_round", 0))
        print(f"[{HOSPITAL_NAME}] Round {round_number}: evaluating global model.")
        self.set_parameters(parameters)
        metrics = evaluate(self.model, self.test_loader)
        print(f"[{HOSPITAL_NAME}] Global model test metrics: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}")
        self.logger.record_evaluate(round_number, metrics)
        return float(metrics["loss"]), len(self.test_loader.dataset), {"accuracy": float(metrics["accuracy"])}


def load_dataset(data_file: str) -> Tuple[DataLoader, DataLoader]:
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing segmented dataset file: {data_file}. Copy it to this directory or update DATA_FILE.")

    data = np.load(data_file)
    x_train = torch.tensor(data["x_train"], dtype=torch.float32).unsqueeze(1) / 255.0
    y_train = torch.tensor(data["y_train"], dtype=torch.long).squeeze()
    x_test = torch.tensor(data["x_test"], dtype=torch.float32).unsqueeze(1) / 255.0
    y_test = torch.tensor(data["y_test"], dtype=torch.long).squeeze()

    print(
        f"[{HOSPITAL_NAME}] Loaded {data_file}: "
        f"train={len(x_train)}, test={len(x_test)}, image_shape={tuple(x_train.shape[1:])}"
    )
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def main() -> None:
    print(f"[{HOSPITAL_NAME}] Connecting to Flower server at {SERVER_ADDRESS}")
    train_loader, test_loader = load_dataset(DATA_FILE)
    model = PneumoniaCNN()
    logger = ClientArtifactLogger()
    client = PneumoniaClient(model, train_loader, test_loader, logger)
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)


if __name__ == "__main__":
    main()
