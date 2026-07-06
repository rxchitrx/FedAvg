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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from compare_runs import collect_run_summaries, write_outputs


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts"
FRONTEND_DIST = ROOT_DIR / "dashboard-ui" / "dist"


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


class ClientConfig(BaseModel):
    hospital_name: str
    data_file: str
    behavior: str = "honest"
    attack_mode: str = "sign_flip"
    attack_strength: float = 4.0
    enabled: bool = True


class RunConfig(BaseModel):
    run_name: str = Field(min_length=1)
    server_address: str = "127.0.0.1:8080"
    num_rounds: int = 5
    local_epochs: int = 2
    aggregation_strategy: str = "fedavg"
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    clients: list[ClientConfig]


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

        common_env = os.environ.copy()
        common_env.update(
            {
                "SERVER_ADDRESS": config.server_address,
                "NUM_ROUNDS": str(config.num_rounds),
                "LOCAL_EPOCHS": str(config.local_epochs),
                "RUN_NAME": config.run_name,
                "ARTIFACT_ROOT": str(ARTIFACT_ROOT),
            }
        )

        server_env = common_env | {"AGGREGATION_STRATEGY": config.aggregation_strategy}
        server = ManagedProcess("server", [sys.executable, "server.py"], server_env)
        server.start()
        self.processes = {"server": server}

        time.sleep(1.0)

        for idx, client in enumerate(config.clients):
            if not client.enabled:
                continue
            client_env = common_env | {
                "HOSPITAL_NAME": client.hospital_name,
                "DATA_FILE": client.data_file,
                "BATCH_SIZE": str(config.batch_size),
                "LEARNING_RATE": str(config.learning_rate),
                "MOMENTUM": str(config.momentum),
                "CLIENT_BEHAVIOR": client.behavior,
                "ATTACK_MODE": client.attack_mode,
                "ATTACK_STRENGTH": str(client.attack_strength),
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

    return {
        "run_name": run_name,
        "server_summary": read_json(server_dir / "summary.json"),
        "fit_rounds": fit_rows,
        "evaluation_rounds": eval_rows,
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
        runs.append(
            {
                "run_name": run_dir.name,
                "aggregation_strategy": metadata.get("aggregation_strategy"),
                "num_rounds": metadata.get("num_rounds"),
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
