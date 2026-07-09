import csv
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from compare_runs import collect_run_summaries, write_outputs


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts"
FRONTEND_DIST = ROOT_DIR / "dashboard-ui" / "dist"
BASE_DATASET_FILES = [ROOT_DIR / "hospital_A_data.npz", ROOT_DIR / "hospital_B_data.npz"]
CASE_DESCRIPTIONS = {
    "different": "Non-IID shards: each client receives a different label/appearance-skewed fragment.",
    "same": "Identical shards: every client receives the exact same local dataset copy.",
    "similar": "Similar fragments: clients receive different stratified fragments from the same combined dataset.",
}


def parse_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def should_keep_log_line(line: str) -> bool:
    noisy_fragments = [
        "DEPRECATED FEATURE",
        "flower-superlink",
        "flower-supernode",
        "Using `start_",
        "This is a deprecated feature.",
        "fork_posix.cc:71",
        "DeprecationWarning:",
        "datetime.datetime.utcnow()",
    ]
    return not any(fragment in line for fragment in noisy_fragments)


class RunConfig(BaseModel):
    run_name: str = Field(min_length=1)
    server_address: str = "127.0.0.1:8080"
    dataset_case: str = "similar"
    client_count: int = Field(default=2, ge=1, le=5)
    num_rounds: int = 5
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9


def load_base_dataset() -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {"x_train": [], "y_train": [], "x_test": [], "y_test": []}
    missing_files = [str(path.name) for path in BASE_DATASET_FILES if not path.exists()]
    if missing_files:
        raise HTTPException(status_code=400, detail=f"Missing base dataset files: {', '.join(missing_files)}")

    for path in BASE_DATASET_FILES:
        data = np.load(path)
        for key in arrays:
            arrays[key].append(data[key])

    return {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}


def save_dataset_shard(path: Path, payload: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train=payload["x_train"],
        y_train=payload["y_train"],
        x_test=payload["x_test"],
        y_test=payload["y_test"],
    )
    y_train = payload["y_train"].reshape(-1)
    y_test = payload["y_test"].reshape(-1)
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    test_labels, test_counts = np.unique(y_test, return_counts=True)
    return {
        "data_file": str(path),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_label_counts": {str(int(label)): int(count) for label, count in zip(train_labels, train_counts)},
        "test_label_counts": {str(int(label)): int(count) for label, count in zip(test_labels, test_counts)},
    }


def stratified_index_splits(labels: np.ndarray, client_count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    splits = [[] for _ in range(client_count)]
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        for client_index, chunk in enumerate(np.array_split(label_indices, client_count)):
            splits[client_index].extend(chunk.tolist())
    return [np.array(sorted(split), dtype=int) for split in splits]


def different_index_splits(images: np.ndarray, labels: np.ndarray, client_count: int) -> list[np.ndarray]:
    brightness = images.reshape(images.shape[0], -1).mean(axis=1)
    order = np.lexsort((brightness, labels.reshape(-1)))
    return [np.array(chunk, dtype=int) for chunk in np.array_split(order, client_count)]


def make_payload(dataset: dict[str, np.ndarray], train_indices: np.ndarray, test_indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "x_train": dataset["x_train"][train_indices],
        "y_train": dataset["y_train"][train_indices],
        "x_test": dataset["x_test"][test_indices],
        "y_test": dataset["y_test"][test_indices],
    }


def prepare_showcase_datasets(config: RunConfig, run_dir: Path) -> list[dict[str, Any]]:
    dataset_case = config.dataset_case.strip().lower()
    if dataset_case not in CASE_DESCRIPTIONS:
        raise HTTPException(status_code=400, detail="Dataset case must be one of: different, same, similar.")

    dataset = load_base_dataset()
    y_train = dataset["y_train"].reshape(-1)
    y_test = dataset["y_test"].reshape(-1)
    dataset_dir = run_dir / "datasets"
    clients: list[dict[str, Any]] = []

    if dataset_case == "same":
        train_splits = [np.arange(len(y_train), dtype=int) for _ in range(config.client_count)]
        test_splits = [np.arange(len(y_test), dtype=int) for _ in range(config.client_count)]
    elif dataset_case == "similar":
        train_splits = stratified_index_splits(y_train, config.client_count, seed=7)
        test_splits = stratified_index_splits(y_test, config.client_count, seed=11)
    else:
        train_splits = different_index_splits(dataset["x_train"], y_train, config.client_count)
        test_splits = different_index_splits(dataset["x_test"], y_test, config.client_count)

    for index in range(config.client_count):
        hospital_name = f"Hospital_{chr(65 + index)}"
        shard_path = dataset_dir / f"{dataset_case}_{hospital_name}.npz"
        stats = save_dataset_shard(shard_path, make_payload(dataset, train_splits[index], test_splits[index]))
        clients.append(
            {
                "hospital_name": hospital_name,
                "data_file": str(shard_path),
                "enabled": True,
                **stats,
            }
        )

    manifest = {
        "dataset_case": dataset_case,
        "description": CASE_DESCRIPTIONS[dataset_case],
        "client_count": config.client_count,
        "source_files": [str(path.name) for path in BASE_DATASET_FILES],
        "clients": clients,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (dataset_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return clients


class ManagedProcess:
    def __init__(self, name: str, command: list[str], env: dict[str, str]) -> None:
        self.name = name
        self.command = command
        self.env = env
        self.logs: deque[str] = deque(maxlen=1000)
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.started_at: float | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command,
            cwd=ROOT_DIR,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.started_at = time.time()
        self.thread = threading.Thread(target=self._drain_logs, daemon=True)
        self.thread.start()

    def _drain_logs(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            clean_line = line.rstrip("\n")
            if should_keep_log_line(clean_line):
                self.logs.append(clean_line)

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)

    def status(self) -> dict[str, Any]:
        code = None if self.process is None else self.process.poll()
        if self.process is None:
            state = "idle"
        elif code is None:
            state = "running"
        elif code == 0:
            state = "finished"
        else:
            state = "failed"
        return {
            "name": self.name,
            "state": state,
            "pid": None if self.process is None else self.process.pid,
            "exit_code": code,
            "uptime_seconds": None if self.started_at is None else round(time.time() - self.started_at, 1),
            "recent_logs": list(self.logs)[-120:],
        }


class RunController:
    def __init__(self) -> None:
        self.processes: dict[str, ManagedProcess] = {}
        self.current_run_name: str | None = None

    def start_run(self, config: RunConfig) -> None:
        if any(proc.status()["state"] == "running" for proc in self.processes.values()):
            raise HTTPException(status_code=400, detail="A run is already active. Stop it before starting another.")

        self.current_run_name = config.run_name
        run_dir = ARTIFACT_ROOT / config.run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        enabled_clients = prepare_showcase_datasets(config, run_dir)

        common_env = os.environ.copy()
        common_env.update(
            {
                "SERVER_ADDRESS": config.server_address,
                "NUM_ROUNDS": str(config.num_rounds),
                "LOCAL_EPOCHS": str(config.local_epochs),
                "RUN_NAME": config.run_name,
                "ARTIFACT_ROOT": str(ARTIFACT_ROOT),
                "MIN_AVAILABLE_CLIENTS": str(len(enabled_clients)),
                "DATASET_CASE": config.dataset_case,
                "CLIENT_COUNT": str(config.client_count),
            }
        )

        server = ManagedProcess("server", [sys.executable, "server.py"], common_env)
        server.start()
        self.processes = {"server": server}

        time.sleep(1.0)

        for idx, client in enumerate(enabled_clients):
            client_env = common_env | {
                "HOSPITAL_NAME": client["hospital_name"],
                "DATA_FILE": client["data_file"],
                "BATCH_SIZE": str(config.batch_size),
                "LEARNING_RATE": str(config.learning_rate),
                "MOMENTUM": str(config.momentum),
            }
            name = f"client_{idx + 1}"
            proc = ManagedProcess(name, [sys.executable, "client.py"], client_env)
            proc.start()
            self.processes[name] = proc
            time.sleep(0.4)

    def stop_run(self) -> None:
        for process in self.processes.values():
            process.stop()

    def process_state(self) -> dict[str, Any]:
        return {name: process.status() for name, process in self.processes.items()}


def collect_run_details(run_name: str) -> dict[str, Any]:
    run_dir = ARTIFACT_ROOT / run_name
    server_dir = run_dir / "server"
    fit_rows = parse_csv(server_dir / "fit_rounds.csv")
    eval_rows = parse_csv(server_dir / "evaluation_rounds.csv")
    server_client_rows = parse_csv(server_dir / "client_rounds.csv")
    dataset_manifest = read_json(run_dir / "datasets" / "manifest.json")
    client_root = run_dir / "clients"
    client_summaries: list[dict[str, Any]] = []
    if client_root.exists():
        for client_dir in sorted(path for path in client_root.iterdir() if path.is_dir()):
            summary = read_json(client_dir / "summary.json")
            round_metrics = read_json(client_dir / "round_metrics.json")
            epoch_rows = parse_csv(client_dir / "epoch_metrics.csv")
            latest_epoch = epoch_rows[-1] if epoch_rows else {}
            client_summaries.append(
                {
                    "name": client_dir.name,
                    "summary": summary,
                    "round_metrics": round_metrics,
                    "latest_epoch": latest_epoch,
                }
            )

    known_client_names = {client["name"] for client in client_summaries}
    server_rows_by_client: dict[str, list[dict[str, Any]]] = {}
    for row in server_client_rows:
        client_name = row.get("hospital_name") or "Unknown_Client"
        server_rows_by_client.setdefault(client_name, []).append(row)

    for client_name, rows in sorted(server_rows_by_client.items()):
        if client_name in known_client_names:
            continue
        latest = rows[-1] if rows else {}
        client_summaries.append(
            {
                "name": client_name,
                "summary": {
                    "metadata": {
                        "hospital_name": client_name,
                        "data_file": latest.get("data_file"),
                    },
                    "latest": latest,
                },
                "round_metrics": {"round_history": rows},
                "latest_epoch": {},
            }
        )

    return {
        "run_name": run_name,
        "server_summary": read_json(server_dir / "summary.json"),
        "fit_rounds": fit_rows,
        "evaluation_rounds": eval_rows,
        "dataset_manifest": dataset_manifest,
        "clients": client_summaries,
    }


def collect_available_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not ARTIFACT_ROOT.exists():
        return runs
    for run_dir in sorted(path for path in ARTIFACT_ROOT.iterdir() if path.is_dir() and path.name != "comparisons"):
        summary = read_json(run_dir / "server" / "summary.json")
        if not summary:
            continue
        metadata = summary.get("metadata", {})
        final_eval = summary.get("final_evaluate") or {}
        dataset_manifest = read_json(run_dir / "datasets" / "manifest.json")
        runs.append(
            {
                "run_name": run_dir.name,
                "aggregation_strategy": metadata.get("aggregation_strategy"),
                "num_rounds": metadata.get("num_rounds"),
                "dataset_case": metadata.get("dataset_case") or dataset_manifest.get("dataset_case"),
                "client_count": metadata.get("client_count") or dataset_manifest.get("client_count"),
                "final_accuracy": final_eval.get("accuracy"),
                "final_recall": final_eval.get("recall"),
                "final_f1_score": final_eval.get("f1_score"),
            }
        )
    return runs


controller = RunController()
app = FastAPI(title="FedAvg Control Center")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    current_run = controller.current_run_name
    comparison_rows = collect_run_summaries()
    return {
        "processes": controller.process_state(),
        "current_run_name": current_run,
        "current_run": collect_run_details(current_run) if current_run else {},
        "available_runs": collect_available_runs(),
        "comparisons": comparison_rows,
    }


@app.post("/api/run/start")
def start_run(config: RunConfig) -> dict[str, Any]:
    controller.start_run(config)
    return {"started": True, "run_name": config.run_name}


@app.post("/api/run/stop")
def stop_run() -> dict[str, bool]:
    controller.stop_run()
    return {"stopped": True}


@app.get("/api/run/{run_name}")
def run_details(run_name: str) -> dict[str, Any]:
    run_dir = ARTIFACT_ROOT / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return collect_run_details(run_name)


@app.post("/api/comparisons/refresh")
def refresh_comparisons() -> dict[str, Any]:
    rows = collect_run_summaries()
    write_outputs(rows)
    return {"rows": rows}


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/")
    def serve_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")
