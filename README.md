# FedAvg

FedAvg is a compact federated learning project that demonstrates how two independent medical institutions can collaboratively train a pneumonia classifier without exchanging raw image data. Each client trains a local PyTorch model on its own partitioned dataset, shares only model weights with a Flower server, and receives an aggregated global model back after each round.

This repository started as a minimal two-file demo and has been upgraded into a more complete project layout with:

- reproducible dependency files
- environment-driven configuration
- a detailed walkthrough for setup and usage
- verbose client-side and server-side training logs
- per-epoch classification metrics during local training
- per-round federated aggregation summaries on the server

## Project Goals

The project is designed to show the core workflow behind Federated Averaging (FedAvg):

1. A central server initializes a shared model.
2. Each client receives the current global model.
3. Each client trains locally on its own data.
4. Each client sends updated weights back to the server.
5. The server computes a weighted average of client updates.
6. The aggregated model is broadcast back to the clients.
7. The process repeats for a fixed number of rounds.

This is not yet a production-grade medical ML system. It is a learning and demo repository focused on showing the mechanics of federated training clearly.

## Repository Layout

```text
FedAvg/
├── client.py               # Flower NumPyClient with verbose local training logs
├── server.py               # Verbose Flower FedAvg server with aggregation logs
├── hospital_A_data.npz     # Local data shard for Hospital A
├── hospital_B_data.npz     # Local data shard for Hospital B
├── README.md               # Project documentation
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Optional developer tooling
├── pyproject.toml          # Project metadata and tool configuration
├── .env.example            # Example runtime configuration
└── .gitignore              # Standard Python ignores
```

## Model Overview

The model is a lightweight convolutional neural network for binary classification:

- input: grayscale `28x28` image
- output classes:
  - `0` = Normal
  - `1` = Pneumonia
- architecture:
  - `Conv2d(1 -> 16)` + `ReLU` + `MaxPool`
  - `Conv2d(16 -> 32)` + `ReLU` + `MaxPool`
  - `Flatten`
  - `Linear(32*7*7 -> 64)` + `ReLU`
  - `Linear(64 -> 2)`

The model is intentionally small so federated rounds are quick to run and easy to inspect from terminal logs.

## Dataset Layout

Each hospital dataset archive contains:

- `x_train`, `y_train`
- `x_val`, `y_val`
- `x_test`, `y_test`

Observed split sizes per hospital:

- train: `2354`
- validation: `524`
- test: `624`

Observed image shape:

- `28x28`
- grayscale
- unsigned 8-bit integer values, normalized in code to `[0, 1]`

## Verbose Logging

One of the main upgrades in this version is detailed terminal output on both sides of the federated run.

### Client-side logs now include

- startup configuration
- dataset summary
- server connection details
- round number
- local training start and completion
- per-epoch training metrics
- per-epoch validation metrics
- parameter export/transmission summary
- server-triggered evaluation metrics on the local test split

Metrics reported include:

- loss
- accuracy
- precision
- recall
- specificity
- F1 score
- confusion-matrix counts: `TP`, `TN`, `FP`, `FN`

### Server-side logs now include

- startup configuration
- round boundaries
- which clients were selected for training
- which clients returned updates
- the per-client metrics received before aggregation
- completion of FedAvg aggregation
- aggregated metric summary across clients
- evaluation dispatch and evaluation result collection
- notification that the global model is being broadcast to clients

This makes the training run much easier to inspect and explain live.

## Configuration

The scripts are configured through environment variables instead of hardcoded edits.

### Shared settings

| Variable | Description | Default |
|---|---|---|
| `SERVER_ADDRESS` | Flower server bind/connect address | `127.0.0.1:8080` |
| `LOCAL_EPOCHS` | Number of local epochs per federated round | `2` |

### Client settings

| Variable | Description | Default |
|---|---|---|
| `HOSPITAL_NAME` | Friendly client name used in logs | `Hospital_A` |
| `DATA_FILE` | Path to local dataset shard | `hospital_A_data.npz` |
| `BATCH_SIZE` | DataLoader batch size | `32` |
| `LEARNING_RATE` | SGD learning rate | `0.01` |
| `MOMENTUM` | SGD momentum | `0.9` |

### Server settings

| Variable | Description | Default |
|---|---|---|
| `NUM_ROUNDS` | Total federated rounds | `5` |
| `MIN_AVAILABLE_CLIENTS` | Minimum clients required to begin a round | `2` |

See [.env.example](/Users/rachit/Code/AIML_federatedLearning/.env.example:1) for a ready-to-copy template.

## Recommended Python Version

Use Python `3.10` to `3.12` for the smoothest compatibility with modern PyTorch and Flower releases. Python `3.14` can be ahead of some ML wheels depending on your environment.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Optional developer tooling:

```bash
pip install -r requirements-dev.txt
```

## How to Run

### 1. Start the server

```bash
export SERVER_ADDRESS="192.168.1.10:8080"
export NUM_ROUNDS=5
export LOCAL_EPOCHS=2
python server.py
```

### 2. Start client A

```bash
export HOSPITAL_NAME="Hospital_A"
export DATA_FILE="hospital_A_data.npz"
export SERVER_ADDRESS="192.168.1.10:8080"
export LOCAL_EPOCHS=2
python client.py
```

### 3. Start client B

```bash
export HOSPITAL_NAME="Hospital_B"
export DATA_FILE="hospital_B_data.npz"
export SERVER_ADDRESS="192.168.1.10:8080"
export LOCAL_EPOCHS=2
python client.py
```

If both clients connect successfully, the server will begin federated rounds once the minimum client count is met.

## Example Training Flow

At a high level, one federated round now looks like this:

1. Server selects both clients for training.
2. Clients receive the current global parameters.
3. Each client trains locally for `LOCAL_EPOCHS`.
4. Each epoch prints:
   - train loss
   - train accuracy / precision / recall / specificity / F1
   - validation loss
   - validation accuracy / precision / recall / specificity / F1
   - confusion counts
5. Each client sends updated weights and final local summary metrics back.
6. Server logs each client result independently.
7. Server performs FedAvg and logs weighted aggregated metrics.
8. Server asks clients to evaluate the aggregated model.
9. Clients return test metrics.
10. Server logs the federated evaluation result.

## Current Limitations

- The project assumes a fixed two-client topology by default.
- There is no experiment tracking backend yet.
- There is no checkpoint persistence yet.
- There are no automated unit or integration tests yet.
- Datasets are committed directly to the repository, which is fine for a demo but not ideal for larger projects.
- Metrics are binary-classification metrics tailored to the current pneumonia task.

## Suggested Next Improvements

If you continue evolving this project, good next upgrades would be:

1. save model checkpoints after each federated round
2. add CSV or JSON training logs
3. add confusion-matrix plots
4. add a simulation mode for running multiple clients on one machine
5. add centralized experiment configuration
6. add tests for metrics, loading, and aggregation helpers
7. separate reusable training code into a `fedavg/` package

## License

No license file has been added yet. If you plan to share or reuse the code publicly, add an explicit license such as MIT.
