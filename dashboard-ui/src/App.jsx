import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const navItems = ["Experiments", "Live Run", "Clients", "Comparisons", "Artifacts", "Settings"];

const defaultConfig = {
  run_name: "dashboard_demo",
  server_address: "127.0.0.1:8080",
  num_rounds: 5,
  local_epochs: 2,
  aggregation_strategy: "fedavg",
  batch_size: 32,
  learning_rate: 0.01,
  momentum: 0.9,
  clients: [
    {
      hospital_name: "Hospital_A",
      data_file: "hospital_A_data.npz",
      behavior: "honest",
      attack_mode: "sign_flip",
      attack_strength: 4,
      enabled: true,
    },
    {
      hospital_name: "Hospital_B",
      data_file: "hospital_B_data.npz",
      behavior: "honest",
      attack_mode: "sign_flip",
      attack_strength: 4,
      enabled: true,
    },
  ],
};

function toNumber(value) {
  if (value === undefined || value === null || value === "") return null;
  const parsed = Number(value);
  return Number.isNaN(parsed) ? null : parsed;
}

function badgeClass(state) {
  return `badge badge-${state || "idle"}`;
}

function App() {
  const [dashboard, setDashboard] = useState({
    processes: {},
    current_run_name: null,
    current_run: {},
    available_runs: [],
    comparisons: [],
  });
  const [selectedRunData, setSelectedRunData] = useState(null);
  const [config, setConfig] = useState(defaultConfig);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedRun, setSelectedRun] = useState("");

  const fetchDashboard = async () => {
    const response = await fetch("/api/dashboard");
    const payload = await response.json();
    setDashboard(payload);
    if (payload.current_run_name) {
      setSelectedRun((current) => current || payload.current_run_name);
    } else if (payload.available_runs?.length) {
      setSelectedRun((current) => current || payload.available_runs[0].run_name);
    }
  };

  useEffect(() => {
    fetchDashboard();
    const id = setInterval(fetchDashboard, 2000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const fetchSelectedRun = async () => {
      if (!selectedRun || selectedRun === dashboard.current_run_name) {
        setSelectedRunData(null);
        return;
      }
      const response = await fetch(`/api/run/${selectedRun}`);
      if (!response.ok) {
        setSelectedRunData(null);
        return;
      }
      setSelectedRunData(await response.json());
    };
    fetchSelectedRun();
  }, [selectedRun, dashboard.current_run_name]);

  const displayedRun = selectedRunData || dashboard.current_run || {};

  const chartData = useMemo(() => {
    const run = displayedRun;
    const fitRounds = run.fit_rounds || [];
    const evalRounds = run.evaluation_rounds || [];
    return fitRounds.map((row) => {
      const matchingEval = evalRounds.find((entry) => entry.round === row.round) || {};
      return {
        round: Number(row.round),
        val_accuracy: toNumber(row.val_accuracy),
        val_recall: toNumber(row.val_recall),
        val_f1_score: toNumber(row.val_f1_score),
        eval_accuracy: toNumber(matchingEval.accuracy),
        eval_recall: toNumber(matchingEval.recall),
        eval_f1_score: toNumber(matchingEval.f1_score),
      };
    });
  }, [displayedRun]);

  const liveTimeline = useMemo(() => {
    const processes = dashboard.processes || {};
    const server = processes.server;
    const clients = Object.values(processes).filter((proc) => proc.name !== "server");
    const displayedEvaluationRounds = displayedRun.evaluation_rounds || [];
    const displayedFitRounds = displayedRun.fit_rounds || [];
    return [
      { label: "Server boot", complete: server?.state === "running" || server?.state === "finished" },
      { label: "Clients connected", complete: clients.length >= 2 || displayedFitRounds.length > 0 },
      { label: "Training active", complete: clients.some((proc) => proc.state === "running") || displayedFitRounds.length > 0 },
      { label: "Aggregation", complete: chartData.length > 0 },
      { label: "Evaluation", complete: displayedEvaluationRounds.length > 0 },
    ];
  }, [dashboard.processes, displayedRun, chartData]);

  const startRun = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/run/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Failed to start run");
      }
      await fetchDashboard();
      setSelectedRun(config.run_name);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const stopRun = async () => {
    setBusy(true);
    setError("");
    try {
      await fetch("/api/run/stop", { method: "POST" });
      await fetchDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const refreshComparisons = async () => {
    await fetch("/api/comparisons/refresh", { method: "POST" });
    await fetchDashboard();
  };

  const updateClient = (index, field, value) => {
    setConfig((current) => {
      const clients = current.clients.map((client, clientIndex) =>
        clientIndex === index ? { ...client, [field]: value } : client
      );
      return { ...current, clients };
    });
  };

  const activeRunName = selectedRun || dashboard.current_run_name;
  const clientCards = (displayedRun.clients || []).slice(0, 2);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">FA</div>
          <div>
            <h1>FedAvg Control Center</h1>
            <p>Evaluator-ready federated learning console</p>
          </div>
        </div>
        <nav className="sidebar-nav">
          {navItems.map((item, index) => (
            <button key={item} className={`nav-item ${index === 0 ? "active" : ""}`}>
              {item}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span className="eyebrow">Current run</span>
          <strong>{activeRunName || "No run selected"}</strong>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar card">
          <div>
            <span className="eyebrow">Experiment control</span>
            <h2>{activeRunName || config.run_name}</h2>
          </div>
          <div className="topbar-controls">
            <div className="inline-field">
              <label>Strategy</label>
              <select
                value={config.aggregation_strategy}
                onChange={(event) =>
                  setConfig((current) => ({ ...current, aggregation_strategy: event.target.value }))
                }
              >
                <option value="fedavg">FedAvg</option>
                <option value="median">Median</option>
              </select>
            </div>
            <button className="primary-button" onClick={startRun} disabled={busy}>
              Start run
            </button>
            <button className="secondary-button" onClick={stopRun} disabled={busy}>
              Stop run
            </button>
            <button className="ghost-button" onClick={refreshComparisons}>
              Refresh comparison
            </button>
          </div>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}

        <section className="grid-two">
          <div className="card config-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Run configuration</span>
                <h3>Scenario builder</h3>
              </div>
            </div>

            <div className="form-grid">
              <label>
                Run name
                <input
                  value={config.run_name}
                  onChange={(event) => setConfig((current) => ({ ...current, run_name: event.target.value }))}
                />
              </label>
              <label>
                Server address
                <input
                  value={config.server_address}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, server_address: event.target.value }))
                  }
                />
              </label>
              <label>
                Rounds
                <input
                  type="number"
                  value={config.num_rounds}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, num_rounds: Number(event.target.value) }))
                  }
                />
              </label>
              <label>
                Local epochs
                <input
                  type="number"
                  value={config.local_epochs}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, local_epochs: Number(event.target.value) }))
                  }
                />
              </label>
              <label>
                Batch size
                <input
                  type="number"
                  value={config.batch_size}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, batch_size: Number(event.target.value) }))
                  }
                />
              </label>
              <label>
                Learning rate
                <input
                  type="number"
                  step="0.001"
                  value={config.learning_rate}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, learning_rate: Number(event.target.value) }))
                  }
                />
              </label>
            </div>

            <div className="client-config-grid">
              {config.clients.map((client, index) => (
                <div className="client-config" key={client.hospital_name}>
                  <div className="client-config-head">
                    <h4>{client.hospital_name}</h4>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={client.enabled}
                        onChange={(event) => updateClient(index, "enabled", event.target.checked)}
                      />
                      Enabled
                    </label>
                  </div>
                  <label>
                    Behavior
                    <select value={client.behavior} onChange={(event) => updateClient(index, "behavior", event.target.value)}>
                      <option value="honest">Honest</option>
                      <option value="poisoned">Poisoned</option>
                    </select>
                  </label>
                  <label>
                    Attack mode
                    <select value={client.attack_mode} onChange={(event) => updateClient(index, "attack_mode", event.target.value)}>
                      <option value="sign_flip">Sign flip</option>
                      <option value="gaussian_noise">Gaussian noise</option>
                      <option value="zero_out">Zero out</option>
                    </select>
                  </label>
                  <label>
                    Attack strength
                    <input
                      type="number"
                      step="0.5"
                      value={client.attack_strength}
                      onChange={(event) => updateClient(index, "attack_strength", Number(event.target.value))}
                    />
                  </label>
                </div>
              ))}
            </div>
          </div>

          <div className="card timeline-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Live progress</span>
                <h3>Run timeline</h3>
              </div>
            </div>
            <div className="timeline">
              {liveTimeline.map((step) => (
                <div key={step.label} className={`timeline-step ${step.complete ? "complete" : ""}`}>
                  <div className="timeline-dot" />
                  <span>{step.label}</span>
                </div>
              ))}
            </div>
            <div className="process-list">
              {Object.values(dashboard.processes || {}).map((process) => (
                <div key={process.name} className="process-row">
                  <div>
                    <strong>{process.name}</strong>
                    <p>{process.pid ? `PID ${process.pid}` : "Not started"}</p>
                  </div>
                  <span className={badgeClass(process.state)}>{process.state}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="content-grid">
          <div className="card chart-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Round metrics</span>
                <h3>Validation and evaluation trends</h3>
              </div>
              <select value={selectedRun} onChange={(event) => setSelectedRun(event.target.value)}>
                <option value="">Current run</option>
                {dashboard.available_runs.map((run) => (
                  <option key={run.run_name} value={run.run_name}>
                    {run.run_name}
                  </option>
                ))}
              </select>
            </div>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#d9e3ea" strokeDasharray="4 4" />
                  <XAxis dataKey="round" stroke="#5f7484" />
                  <YAxis stroke="#5f7484" domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="eval_accuracy" stroke="#1b5e74" strokeWidth={3} dot={false} name="Eval accuracy" />
                  <Line type="monotone" dataKey="eval_recall" stroke="#0d9488" strokeWidth={3} dot={false} name="Eval recall" />
                  <Line type="monotone" dataKey="eval_f1_score" stroke="#d97706" strokeWidth={3} dot={false} name="Eval F1" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="clients-column">
            {clientCards.map((client) => {
              const latest = client.latest_epoch || {};
              const metadata = client.summary?.metadata || {};
              return (
                <div key={client.name} className="card client-card">
                  <div className="section-head">
                    <div>
                      <span className="eyebrow">Client status</span>
                      <h3>{client.name}</h3>
                    </div>
                    <span className={badgeClass(metadata.client_behavior)}>{metadata.client_behavior || "unknown"}</span>
                  </div>
                  <div className="metric-pair-grid">
                    <div>
                      <span>Train acc</span>
                      <strong>{latest.train_accuracy ? Number(latest.train_accuracy).toFixed(3) : "--"}</strong>
                    </div>
                    <div>
                      <span>Val acc</span>
                      <strong>{latest.val_accuracy ? Number(latest.val_accuracy).toFixed(3) : "--"}</strong>
                    </div>
                    <div>
                      <span>Train recall</span>
                      <strong>{latest.train_recall ? Number(latest.train_recall).toFixed(3) : "--"}</strong>
                    </div>
                    <div>
                      <span>Val F1</span>
                      <strong>{latest.val_f1_score ? Number(latest.val_f1_score).toFixed(3) : "--"}</strong>
                    </div>
                  </div>
                  <div className="confusion-note">
                    <span>Attack mode</span>
                    <strong>{metadata.attack_mode || "none"}</strong>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="grid-two">
          <div className="card comparison-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Run comparison</span>
                <h3>Final outcomes</h3>
              </div>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Strategy</th>
                  <th>Accuracy</th>
                  <th>Recall</th>
                  <th>F1</th>
                </tr>
              </thead>
              <tbody>
                {dashboard.comparisons.map((row) => {
                  const best = dashboard.comparisons.reduce(
                    (currentBest, item) =>
                      (item.final_eval_f1_score ?? -1) > (currentBest.final_eval_f1_score ?? -1) ? item : currentBest,
                    dashboard.comparisons[0] || {}
                  );
                  const isBest = best.run_name === row.run_name;
                  return (
                    <tr key={row.run_name} className={isBest ? "best-row" : ""}>
                      <td>{row.run_name}</td>
                      <td>{row.aggregation_strategy}</td>
                      <td>{row.final_eval_accuracy ? Number(row.final_eval_accuracy).toFixed(3) : "--"}</td>
                      <td>{row.final_eval_recall ? Number(row.final_eval_recall).toFixed(3) : "--"}</td>
                      <td>{row.final_eval_f1_score ? Number(row.final_eval_f1_score).toFixed(3) : "--"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="card logs-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Process logs</span>
                <h3>Readable activity feed</h3>
              </div>
            </div>
            <div className="logs-grid">
              {Object.values(dashboard.processes || {}).map((process) => (
                <div key={process.name} className="log-column">
                  <div className="log-column-head">
                    <strong>{process.name}</strong>
                    <span className={badgeClass(process.state)}>{process.state}</span>
                  </div>
                  <pre>{(process.recent_logs || []).slice(-12).join("\n") || "No logs yet."}</pre>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
