# FedAvg

FedAvg is a simple federated learning demo for binary pneumonia detection. One Flower server coordinates multiple hospital clients. Each client trains a small PyTorch CNN on its own local `.npz` dataset shard, sends model parameters to the server, and receives the averaged global model back after each round.

This version intentionally keeps the architecture focused:

- standard FedAvg only
- server and clients run locally from the dashboard on one laptop
- dashboard supports 1 to 5 clients
- dashboard can generate three dataset-case demos: different, same, and similar fragments

## Project Layout

```text
FedAvg/
├── client.py                  # Flower client: local training + evaluation
├── server.py                  # Flower server: standard FedAvg
├── dashboard_api.py           # FastAPI dashboard controller
├── dashboard-ui/              # React dashboard
├── compare_runs.py            # Saved-run comparison helper
├── hospital_A_data.npz        # Hospital A dataset shard
├── hospital_B_data.npz        # Hospital B dataset shard
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── .env.example
```

## Dataset

Each hospital file contains:

- `x_train`, `y_train`
- `x_test`, `y_test`
- optional validation arrays may exist, but the basic client uses train and test

Current local shards:

| File | Train samples | Test samples |
|---|---:|---:|
| `hospital_A_data.npz` | 2,354 | 624 |
| `hospital_B_data.npz` | 2,354 | 624 |

Images are `28x28` grayscale. Labels are:

- `0` = Normal
- `1` = Pneumonia

## Model

The clients train the same lightweight CNN:

- `Conv2d(1 -> 16)` + ReLU + MaxPool
- `Conv2d(16 -> 32)` + ReLU + MaxPool
- flatten
- `Linear(32*7*7 -> 64)` + ReLU
- `Linear(64 -> 2)`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install dashboard dependencies:

```bash
cd dashboard-ui
npm install
npm run build
cd ..
```

## Run From Terminal

The dashboard is the recommended way to run the final demo because it generates the selected dataset case automatically. Terminal mode is still available for quick manual testing with existing `.npz` files.

### 1. Start the server

```bash
source .venv/bin/activate
export SERVER_ADDRESS="127.0.0.1:8080"
export NUM_ROUNDS=5
export MIN_AVAILABLE_CLIENTS=2
export LOCAL_EPOCHS=2
export RUN_NAME="fedavg_demo"
python server.py
```

### 2. Start Hospital A client

```bash
source .venv/bin/activate
export HOSPITAL_NAME="Hospital_A"
export DATA_FILE="hospital_A_data.npz"
export SERVER_ADDRESS="127.0.0.1:8080"
export RUN_NAME="fedavg_demo"
python client.py
```

### 3. Start Hospital B client

```bash
source .venv/bin/activate
export HOSPITAL_NAME="Hospital_B"
export DATA_FILE="hospital_B_data.npz"
export SERVER_ADDRESS="127.0.0.1:8080"
export RUN_NAME="fedavg_demo"
python client.py
```

## Dashboard

The dashboard is a presenter-friendly control panel for the same basic workflow.

Start it with:

```bash
source .venv/bin/activate
uvicorn dashboard_api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The dashboard shows:

- local server address and run configuration
- selected dataset case
- selected client count from 1 to 5
- generated client dataset shards
- live process status
- local client training metrics
- number of tensors/scalars sent by each client
- FedAvg aggregation progress
- global model accuracy/loss by round
- a presentable log explaining what happened in the run

## Dashboard Showcase Cases

The dashboard generates per-run dataset shards under `artifacts/<RUN_NAME>/datasets/` from the available base files.

| Case | What It Shows | How Shards Are Generated |
|---|---|---|
| `Different datasets` | FedAvg under strongly non-IID clients | clients receive different label/appearance-skewed fragments |
| `Same dataset` | FedAvg when every client has identical data | every client receives the exact same combined dataset copy |
| `Similar fragments` | FedAvg under normal IID-like fragments | clients receive different stratified fragments with similar label ratios |

The `Number of clients` dropdown supports `1` through `5`. With one client, the run acts as a local-training baseline through the same server/client protocol.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `SERVER_ADDRESS` | Local Flower server address | `127.0.0.1:8080` |
| `NUM_ROUNDS` | Federated communication rounds | `5` |
| `MIN_AVAILABLE_CLIENTS` | Required clients before server starts a round | `2` |
| `DATASET_CASE` | Dashboard-generated dataset case label | `manual` |
| `CLIENT_COUNT` | Number of configured local clients | `MIN_AVAILABLE_CLIENTS` |
| `LOCAL_EPOCHS` | Local epochs per round | `2` |
| `RUN_NAME` | Artifact folder name | `default_run` |
| `ARTIFACT_ROOT` | Root output directory | `artifacts` |
| `HOSPITAL_NAME` | Client display name | `Hospital_A` |
| `DATA_FILE` | Client dataset shard | `hospital_A_data.npz` |
| `BATCH_SIZE` | Client batch size | `32` |
| `LEARNING_RATE` | SGD learning rate | `0.01` |
| `MOMENTUM` | SGD momentum | `0.9` |

## Artifacts

Each run writes outputs under:

```text
artifacts/<RUN_NAME>/
```

Server artifacts:

- `datasets/manifest.json`
- `datasets/<case>_<HOSPITAL_NAME>.npz`
- `server/fit_rounds.csv`
- `server/evaluation_rounds.csv`
- `server/client_rounds.csv`
- `server/summary.json`

Client artifacts:

- `clients/<HOSPITAL_NAME>/epoch_metrics.csv`
- `clients/<HOSPITAL_NAME>/round_metrics.json`
- `clients/<HOSPITAL_NAME>/summary.json`

Compare saved runs:

```bash
python compare_runs.py
```
