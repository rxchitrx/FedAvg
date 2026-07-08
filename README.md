# FedAvg

FedAvg is a simple federated learning demo for binary pneumonia detection. One Flower server coordinates multiple hospital clients. Each client trains a small PyTorch CNN on its own local `.npz` dataset shard, sends model parameters to the server, and receives the averaged global model back after each round.

This version intentionally keeps the architecture focused:

- standard FedAvg only
- clients can connect to the server using an IP address and port
- the dashboard only visualizes this basic FedAvg workflow

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

### 1. Start the server

For local-only testing:

```bash
source .venv/bin/activate
export SERVER_ADDRESS="127.0.0.1:8080"
export NUM_ROUNDS=5
export MIN_AVAILABLE_CLIENTS=2
export LOCAL_EPOCHS=2
export RUN_NAME="fedavg_demo"
python server.py
```

For other laptops/devices to connect, bind to all network interfaces:

```bash
export SERVER_ADDRESS="0.0.0.0:8080"
python server.py
```

Other clients should connect using your machine's reachable IP, for example:

```bash
export SERVER_ADDRESS="192.168.1.24:8080"
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

- server address and run configuration
- enabled clients and their dataset shards
- live process status
- local client training metrics
- number of tensors/scalars sent by each client
- FedAvg aggregation progress
- global model accuracy/loss by round
- a presentable log explaining what happened in the run

## Multi-Laptop Demo

Use this mode when one laptop is the central server and other laptops act as hospital clients.

### 1. Put every laptop on the same network

Use the same Wi-Fi or the same mobile hotspot. Find the server laptop IP:

```bash
ipconfig getifaddr en0
```

If that prints nothing, try:

```bash
ipconfig getifaddr en1
```

In the examples below, replace `192.168.1.24` with the real server laptop IP.

### 2. Start the dashboard on the server laptop

```bash
source .venv/bin/activate
uvicorn dashboard_api:app --host 0.0.0.0 --port 8000
```

Open the dashboard on the server laptop:

```text
http://127.0.0.1:8000
```

Other laptops can also view it at:

```text
http://192.168.1.24:8000
```

### 3. Configure the run in the dashboard

Set:

- `Server bind address`: `0.0.0.0:8080`
- `Client connection address`: `192.168.1.24:8080`
- uncheck `Launch clients on this laptop automatically`
- keep `Hospital_A` and `Hospital_B` enabled, or enable only the clients that will actually connect
- use `Add client` for extra laptops such as `Hospital_C`, `Hospital_D`, and point each one at its own dataset shard

Click `Start FedAvg run`. The server laptop will start the Flower server and wait for the remote clients.

### 4. Prepare each client laptop

On each client laptop:

```bash
git clone https://github.com/rxchitrx/FedAvg.git
cd FedAvg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure the correct dataset shard is present on that laptop:

- Hospital A laptop needs `hospital_A_data.npz`
- Hospital B laptop needs `hospital_B_data.npz`

### 5. Start Hospital A from a client laptop

```bash
source .venv/bin/activate
export HOSPITAL_NAME="Hospital_A"
export DATA_FILE="hospital_A_data.npz"
export SERVER_ADDRESS="192.168.1.24:8080"
export RUN_NAME="fedavg_demo"
export LOCAL_EPOCHS=2
export BATCH_SIZE=32
export LEARNING_RATE=0.01
export MOMENTUM=0.9
python client.py
```

### 6. Start Hospital B from another client laptop

```bash
source .venv/bin/activate
export HOSPITAL_NAME="Hospital_B"
export DATA_FILE="hospital_B_data.npz"
export SERVER_ADDRESS="192.168.1.24:8080"
export RUN_NAME="fedavg_demo"
export LOCAL_EPOCHS=2
export BATCH_SIZE=32
export LEARNING_RATE=0.01
export MOMENTUM=0.9
python client.py
```

The dashboard also prints these commands for the enabled clients when remote-client mode is selected.

For more than two laptops, click `Add client` in the dashboard, give each client a unique hospital name, and make sure each client laptop runs the generated command with its matching `HOSPITAL_NAME` and `DATA_FILE`.

If a client cannot connect, check that:

- all laptops are on the same network
- the server laptop firewall allows incoming connections
- port `8080` is not already in use
- each client uses the server laptop IP, not `127.0.0.1`

## Configuration

| Variable | Description | Default |
|---|---|---|
| `SERVER_ADDRESS` | Server bind/connect address | `127.0.0.1:8080` for client, `0.0.0.0:8080` for server |
| `NUM_ROUNDS` | Federated communication rounds | `5` |
| `MIN_AVAILABLE_CLIENTS` | Required clients before server starts a round | `2` |
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
