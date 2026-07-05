import logging
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("fedavg-client")


# Runtime configuration
HOSPITAL_NAME = os.getenv("HOSPITAL_NAME", "Hospital_A")
DATA_FILE = os.getenv("DATA_FILE", "hospital_A_data.npz")
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))
MOMENTUM = float(os.getenv("MOMENTUM", "0.9"))


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


def summarize_parameters(parameters: List[np.ndarray]) -> str:
    total_scalars = sum(int(np.prod(layer.shape)) for layer in parameters)
    return f"{len(parameters)} tensors | {total_scalars:,} scalar values"


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_binary_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)

    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    predictions: List[int] = []
    labels: List[int] = []

    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)

            batch_predictions = torch.argmax(outputs, dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    metrics = compute_binary_metrics(predictions, labels)
    metrics["loss"] = safe_divide(total_loss, len(loader.dataset))
    return metrics


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    predictions: List[int] = []
    labels: List[int] = []

    for images, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_labels.size(0)
        batch_predictions = torch.argmax(outputs, dim=1)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        labels.extend(batch_labels.cpu().tolist())

    metrics = compute_binary_metrics(predictions, labels)
    metrics["loss"] = safe_divide(total_loss, len(train_loader.dataset))
    return metrics


def log_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    LOGGER.info(
        "%s loss=%.4f acc=%.4f precision=%.4f recall=%.4f specificity=%.4f f1=%.4f "
        "tp=%d tn=%d fp=%d fn=%d",
        prefix,
        metrics["loss"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["specificity"],
        metrics["f1_score"],
        int(metrics["tp"]),
        int(metrics["tn"]),
        int(metrics["fp"]),
        int(metrics["fn"]),
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate_loader(model, val_loader, criterion)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_specificity": train_metrics["specificity"],
            "train_f1_score": train_metrics["f1_score"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_f1_score": val_metrics["f1_score"],
        }
        history.append(epoch_metrics)

        LOGGER.info("[%s] Epoch %d/%d completed", HOSPITAL_NAME, epoch, epochs)
        log_metrics(f"[{HOSPITAL_NAME}]   Train:", train_metrics)
        log_metrics(f"[{HOSPITAL_NAME}]   Valid:", val_metrics)

    return history, history[-1]


class PneumoniaClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        parameters = [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]
        LOGGER.info(
            "[%s] Exporting local model parameters to Flower server: %s",
            HOSPITAL_NAME,
            summarize_parameters(parameters),
        )
        return parameters

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        round_number = int(config.get("server_round", 0))
        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        LOGGER.info(
            "[%s] Round %d | received global parameters from server, starting local training for %d epoch(s)",
            HOSPITAL_NAME,
            round_number,
            local_epochs,
        )

        self.set_parameters(parameters)
        history, final_epoch_metrics = train(self.model, self.train_loader, self.val_loader, epochs=local_epochs)
        updated_parameters = self.get_parameters(config={})

        LOGGER.info(
            "[%s] Round %d | local training finished, transmitting updated parameters back to server",
            HOSPITAL_NAME,
            round_number,
        )
        LOGGER.info(
            "[%s] Round %d | transmission summary: train_examples=%d, validation_examples=%d, %s",
            HOSPITAL_NAME,
            round_number,
            len(self.train_loader.dataset),
            len(self.val_loader.dataset),
            summarize_parameters(updated_parameters),
        )

        return updated_parameters, len(self.train_loader.dataset), {
            "round": float(round_number),
            "epochs_completed": float(len(history)),
            "train_loss": final_epoch_metrics["train_loss"],
            "train_accuracy": final_epoch_metrics["train_accuracy"],
            "train_precision": final_epoch_metrics["train_precision"],
            "train_recall": final_epoch_metrics["train_recall"],
            "train_specificity": final_epoch_metrics["train_specificity"],
            "train_f1_score": final_epoch_metrics["train_f1_score"],
            "val_loss": final_epoch_metrics["val_loss"],
            "val_accuracy": final_epoch_metrics["val_accuracy"],
            "val_precision": final_epoch_metrics["val_precision"],
            "val_recall": final_epoch_metrics["val_recall"],
            "val_specificity": final_epoch_metrics["val_specificity"],
            "val_f1_score": final_epoch_metrics["val_f1_score"],
        }

    def evaluate(self, parameters, config):
        round_number = int(config.get("server_round", 0))
        LOGGER.info("[%s] Round %d | evaluation request received from server", HOSPITAL_NAME, round_number)
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        metrics = evaluate_loader(self.model, self.test_loader, criterion)
        log_metrics(f"[{HOSPITAL_NAME}] Round {round_number} | Test:", metrics)

        return float(metrics["loss"]), len(self.test_loader.dataset), {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1_score": metrics["f1_score"],
        }


def load_dataset_split(data_file: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Missing segmented dataset file: {data_file}. Place it in the project root or update DATA_FILE."
        )

    LOGGER.info("[%s] Loading local dataset from %s", HOSPITAL_NAME, data_file)
    data = np.load(data_file)

    x_train = torch.tensor(data["x_train"], dtype=torch.float32).unsqueeze(1) / 255.0
    y_train = torch.tensor(data["y_train"], dtype=torch.long).squeeze()
    x_val = torch.tensor(data["x_val"], dtype=torch.float32).unsqueeze(1) / 255.0
    y_val = torch.tensor(data["y_val"], dtype=torch.long).squeeze()
    x_test = torch.tensor(data["x_test"], dtype=torch.float32).unsqueeze(1) / 255.0
    y_test = torch.tensor(data["y_test"], dtype=torch.long).squeeze()

    LOGGER.info(
        "[%s] Dataset summary | train=%d val=%d test=%d image_shape=%s batch_size=%d",
        HOSPITAL_NAME,
        len(x_train),
        len(x_val),
        len(x_test),
        tuple(x_train.shape[1:]),
        BATCH_SIZE,
    )

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def main() -> None:
    LOGGER.info("[%s] Client bootstrapping started", HOSPITAL_NAME)
    LOGGER.info(
        "[%s] Runtime config | server_address=%s data_file=%s local_epochs=%d learning_rate=%.4f momentum=%.2f",
        HOSPITAL_NAME,
        SERVER_ADDRESS,
        DATA_FILE,
        LOCAL_EPOCHS,
        LEARNING_RATE,
        MOMENTUM,
    )

    train_loader, val_loader, test_loader = load_dataset_split(DATA_FILE)
    model = PneumoniaCNN()
    client = PneumoniaClient(model, train_loader, val_loader, test_loader)

    LOGGER.info("[%s] Opening Flower client connection to %s", HOSPITAL_NAME, SERVER_ADDRESS)
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)


if __name__ == "__main__":
    main()
