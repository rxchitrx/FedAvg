# FedAvg

FedAvg is a compact federated learning project for privacy-preserving pneumonia detection. Two hospital clients train a shared PyTorch model locally on their own chest X-ray partitions, send model updates to a Flower server, and receive an aggregated global model back after each communication round.

This repository is now set up as an evaluator-ready prototype with:

- project metadata and dependency files
- verbose client/server logs
- malicious-client simulation mode
- switchable `fedavg` and robust `median` aggregation
- CSV/JSON artifact logging
- post-run comparison output for presentation evidence

## Evaluator Demo Goal

The intended final-demo story is:

1. show privacy-preserving collaboration across hospitals
2. show standard FedAvg training in a clean setting
3. show how a poisoned client can hurt standard averaging
4. show how a robust median aggregator resists that poisoned update better
5. show all results again through saved CSV/JSON outputs

## Repository Layout

```text
FedAvg/
├── client.py               # Flower client with verbose local training and attack mode
├── server.py               # Flower server with FedAvg/median strategy selection
├── compare_runs.py         # Compare completed experiment runs from saved artifacts
├── hospital_A_data.npz     # Local dataset shard for Hospital A
├── hospital_B_data.npz     # Local dataset shard for Hospital B
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Model Overview

The project uses a lightweight CNN for binary classification:

- input: grayscale `28x28` chest X-ray image
- labels:
  - `0` = Normal
  - `1` = Pneumonia
- architecture:
  - `Conv2d(1 -> 16)` + `ReLU` + `MaxPool`
  - `Conv2d(16 -> 32)` + `ReLU` + `MaxPool`
  - `Flatten`
  - `Linear(32*7*7 -> 64)` + `ReLU`
  - `Linear(64 -> 2)`

The model is intentionally small so training rounds are fast and easy to demonstrate live.

## Dataset Layout

Each hospital archive contains:

- `x_train`, `y_train`
- `x_val`, `y_val`
- `x_test`, `y_test`

Observed per-hospital split sizes:

- train: `2354`
- validation: `524`
- test: `624`

Observed image format:

- `28x28`
- grayscale
- uint8 in storage, normalized to `[0, 1]` in code

## Verbose Logging

### Client logs include

- startup configuration
- dataset summary
- current round number
- per-epoch train metrics
- per-epoch validation metrics
- outgoing parameter transmission summary
- test metrics during server-triggered evaluation
- warning banner when malicious mode is enabled

### Server logs include

- runtime configuration
- selected clients per round
- per-client fit metrics received
- aggregation strategy used
- aggregated round summary
- evaluation request dispatch
- per-client evaluation results
- final aggregated evaluation metrics

### Metrics reported

- loss
- accuracy
- precision
- recall
- specificity
- F1 score
- confusion counts: `TP`, `TN`, `FP`, `FN`

## Attack Simulation

Client behavior is configurable with environment variables.

Set a client to honest mode:

```bash
export CLIENT_BEHAVIOR=honest
```

Set a client to malicious mode:

```bash
export CLIENT_BEHAVIOR=poisoned
```

Supported `ATTACK_MODE` values:

- `sign_flip`
- `gaussian_noise`
- `zero_out`

Supported attack settings:

- `ATTACK_MODE`
- `ATTACK_STRENGTH`

In malicious mode, the client trains normally but tampers with the outgoing update before it is transmitted to the server. That makes the attack visible in both the logs and the final comparison outputs.

## Aggregation Strategies

Set the server aggregation strategy with:

```bash
export AGGREGATION_STRATEGY=fedavg
```

or

```bash
export AGGREGATION_STRATEGY=median
```

### `fedavg`

Standard weighted federated averaging.

### `median`

Coordinate-wise median aggregation across client updates. This is a stronger demo baseline against extreme poisoned updates than plain FedAvg.

## Artifact Logging

Every run writes structured outputs under:

```text
artifacts/<RUN_NAME>/
```

### Client artifacts

Each client writes:

- `epoch_metrics.csv`
- `round_metrics.json`
- `summary.json`

### Server artifacts

The server writes:

- `fit_rounds.csv`
- `evaluation_rounds.csv`
- `summary.json`

### Run comparison artifacts

After multiple runs:

```bash
python compare_runs.py
```

This generates:

- `artifacts/comparisons/run_comparison.json`
- `artifacts/comparisons/run_comparison.csv`

and prints a simple comparison table in the terminal.

## Configuration

### Shared settings

| Variable | Description | Default |
|---|---|---|
| `SERVER_ADDRESS` | Flower server bind/connect address | `127.0.0.1:8080` |
| `LOCAL_EPOCHS` | Local epochs per federated round | `2` |
| `RUN_NAME` | Artifact folder name for the current experiment | `default_run` |
| `ARTIFACT_ROOT` | Root directory for saved outputs | `artifacts` |

### Client settings

| Variable | Description | Default |
|---|---|---|
| `HOSPITAL_NAME` | Friendly client label in logs | `Hospital_A` |
| `DATA_FILE` | Local dataset shard file | `hospital_A_data.npz` |
| `BATCH_SIZE` | DataLoader batch size | `32` |
| `LEARNING_RATE` | SGD learning rate | `0.01` |
| `MOMENTUM` | SGD momentum | `0.9` |
| `CLIENT_BEHAVIOR` | `honest` or `poisoned` | `honest` |
| `ATTACK_MODE` | `sign_flip`, `gaussian_noise`, or `zero_out` | `sign_flip` |
| `ATTACK_STRENGTH` | Attack intensity | `4.0` |

### Server settings

| Variable | Description | Default |
|---|---|---|
| `NUM_ROUNDS` | Total federated rounds | `5` |
| `MIN_AVAILABLE_CLIENTS` | Minimum required clients | `2` |
| `AGGREGATION_STRATEGY` | `fedavg` or `median` | `fedavg` |

See [.env.example](/Users/rachit/Code/AIML_federatedLearning/.env.example:1) for a copyable template.

## Recommended Python Version

Use Python `3.10` to `3.12` if possible for the smoothest PyTorch/Flower compatibility.

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Optional development tooling:

```bash
pip install -r requirements-dev.txt
```

## Demo Scenarios

### 1. Clean baseline with FedAvg

Server:

```bash
export SERVER_ADDRESS="127.0.0.1:8080"
export NUM_ROUNDS=5
export LOCAL_EPOCHS=2
export RUN_NAME="baseline_clean_fedavg"
export AGGREGATION_STRATEGY="fedavg"
python server.py
```

Client A:

```bash
export HOSPITAL_NAME="Hospital_A"
export DATA_FILE="hospital_A_data.npz"
export SERVER_ADDRESS="127.0.0.1:8080"
export LOCAL_EPOCHS=2
export RUN_NAME="baseline_clean_fedavg"
export CLIENT_BEHAVIOR="honest"
python client.py
```

Client B:

```bash
export HOSPITAL_NAME="Hospital_B"
export DATA_FILE="hospital_B_data.npz"
export SERVER_ADDRESS="127.0.0.1:8080"
export LOCAL_EPOCHS=2
export RUN_NAME="baseline_clean_fedavg"
export CLIENT_BEHAVIOR="honest"
python client.py
```

### 2. Attacked baseline with FedAvg

Server:

```bash
export RUN_NAME="attack_fedavg"
export AGGREGATION_STRATEGY="fedavg"
python server.py
```

Honest client:

```bash
export HOSPITAL_NAME="Hospital_A"
export DATA_FILE="hospital_A_data.npz"
export RUN_NAME="attack_fedavg"
export CLIENT_BEHAVIOR="honest"
python client.py
```

Poisoned client:

```bash
export HOSPITAL_NAME="Hospital_B"
export DATA_FILE="hospital_B_data.npz"
export RUN_NAME="attack_fedavg"
export CLIENT_BEHAVIOR="poisoned"
export ATTACK_MODE="sign_flip"
export ATTACK_STRENGTH="4.0"
python client.py
```

### 3. Defended run with median aggregation

Server:

```bash
export RUN_NAME="attack_median"
export AGGREGATION_STRATEGY="median"
python server.py
```

Honest client:

```bash
export HOSPITAL_NAME="Hospital_A"
export DATA_FILE="hospital_A_data.npz"
export RUN_NAME="attack_median"
export CLIENT_BEHAVIOR="honest"
python client.py
```

Poisoned client:

```bash
export HOSPITAL_NAME="Hospital_B"
export DATA_FILE="hospital_B_data.npz"
export RUN_NAME="attack_median"
export CLIENT_BEHAVIOR="poisoned"
export ATTACK_MODE="sign_flip"
export ATTACK_STRENGTH="4.0"
python client.py
```

### 4. Compare completed runs

```bash
python compare_runs.py
```

This produces evaluator-friendly summary files and a terminal table.

## Suggested Presentation Evidence

For your final review, the strongest proof will be:

1. one screenshot of verbose honest-client training logs
2. one screenshot of the poisoned-client warning and transmission
3. one screenshot of server aggregation logs under `fedavg`
4. one screenshot of server aggregation logs under `median`
5. one table from `compare_runs.py`
6. one slide showing the saved artifact folder structure

## Current Limitations

- The project still assumes a small fixed client topology by default.
- Coordinate-wise median is a useful robust baseline, not a complete security framework.
- There is no checkpoint persistence yet.
- There are no automated tests yet.
- The datasets are committed directly into the repo for demo convenience.

## Good Next Improvements

1. add global-model checkpoint saving per round
2. add confusion-matrix plots for presentation slides
3. add a single-machine orchestration script to launch server and clients together
4. add more attack types and stronger robust aggregators
5. add tests for metric calculations and aggregation behavior

## License

No license file has been added yet. If you plan to share or reuse the code publicly, add an explicit license such as MIT.
