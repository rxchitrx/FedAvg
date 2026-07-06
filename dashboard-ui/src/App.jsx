import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const navItems = [
  { id: "experiments", label: "Experiments" },
  { id: "live-run", label: "Live Run" },
  { id: "clients", label: "Clients" },
  { id: "comparisons", label: "Comparisons" },
  { id: "artifacts", label: "Artifacts" },
  { id: "settings", label: "Settings" },
];

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

function formatMetric(value, digits = 3) {
  const number = toNumber(value);
  return number === null ? "--" : number.toFixed(digits);
}

function badgeClass(state) {
  return `badge badge-${state || "idle"}`;
}

function getLatestByPhase(roundHistory, phase) {
  return [...(roundHistory || [])].reverse().find((entry) => entry.phase === phase) || null;
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
  const [activeSection, setActiveSection] = useState("experiments");
  const [chartMetric, setChartMetric] = useState("accuracy");

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
    const fitRounds = displayedRun.fit_rounds || [];
    const evalRounds = displayedRun.evaluation_rounds || [];
    const latestFitByRound = new Map();
    const latestEvalByRound = new Map();

    fitRounds.forEach((row) => {
      const round = toNumber(row.round);
      if (round !== null) latestFitByRound.set(round, row);
    });
    evalRounds.forEach((row) => {
      const round = toNumber(row.round);
      if (round !== null) latestEvalByRound.set(round, row);
    });

    return Array.from(latestFitByRound.keys())
      .sort((a, b) => a - b)
      .map((round) => {
        const fitRow = latestFitByRound.get(round) || {};
        const evalRow = latestEvalByRound.get(round) || {};
        return {
          round,
          local_accuracy: toNumber(fitRow.local_test_accuracy ?? fitRow.val_accuracy),
          local_recall: toNumber(fitRow.local_test_recall ?? fitRow.val_recall),
          local_f1_score: toNumber(fitRow.local_test_f1_score ?? fitRow.val_f1_score),
          global_accuracy: toNumber(evalRow.accuracy),
          global_recall: toNumber(evalRow.recall),
          global_f1_score: toNumber(evalRow.f1_score),
        };
      });
  }, [displayedRun]);

  const chartConfig = {
    accuracy: {
      title: "Accuracy by round",
      localKey: "local_accuracy",
      globalKey: "global_accuracy",
      localLabel: "Local test accuracy",
      globalLabel: "Global test accuracy",
    },
    recall: {
      title: "Recall by round",
      localKey: "local_recall",
      globalKey: "global_recall",
      localLabel: "Local test recall",
      globalLabel: "Global test recall",
    },
    f1_score: {
      title: "F1 by round",
      localKey: "local_f1_score",
      globalKey: "global_f1_score",
      localLabel: "Local test F1",
      globalLabel: "Global test F1",
    },
  };

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
      { label: "Aggregation", complete: displayedFitRounds.length > 0 },
      { label: "Evaluation", complete: displayedEvaluationRounds.length > 0 },
    ];
  }, [dashboard.processes, displayedRun]);

  const clientCards = useMemo(() => {
    return (displayedRun.clients || []).map((client) => {
      const roundHistory = client.round_metrics?.round_history || [];
      return {
        ...client,
        latestFit: getLatestByPhase(roundHistory, "fit"),
        latestEvaluate: getLatestByPhase(roundHistory, "evaluate"),
      };
    });
  }, [displayedRun]);

  const globalSummary = useMemo(() => {
    const summary = displayedRun.server_summary || {};
    return {
      finalFit: summary.final_fit || null,
      finalEvaluate: summary.final_evaluate || null,
      metadata: summary.metadata || {},
    };
  }, [displayedRun]);

  const processLogs = dashboard.processes || {};

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
      setActiveSection("live-run");
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

  const jumpToSection = (sectionId) => {
    setActiveSection(sectionId);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const activeRunName = selectedRun || dashboard.current_run_name;

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
          {navItems.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${activeSection === item.id ? "active" : ""}`}
              onClick={() => jumpToSection(item.id)}
            >
              {item.label}
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

        <section id="experiments" className="grid-two section-block">
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
                  <div className="behavior-group">
                    <span>Behavior</span>
                    <div className="segmented-control">
                      <button
                        type="button"
                        className={client.behavior === "honest" ? "segment active" : "segment"}
                        onClick={() => updateClient(index, "behavior", "honest")}
                      >
                        Honest
                      </button>
                      <button
                        type="button"
                        className={client.behavior === "poisoned" ? "segment danger active" : "segment danger"}
                        onClick={() => updateClient(index, "behavior", "poisoned")}
                      >
                        Poisoned
                      </button>
                    </div>
                  </div>
                  <label>
                    Attack mode
                    <select
                      value={client.attack_mode}
                      disabled={client.behavior !== "poisoned"}
                      onChange={(event) => updateClient(index, "attack_mode", event.target.value)}
                    >
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
                      disabled={client.behavior !== "poisoned"}
                      value={client.attack_strength}
                      onChange={(event) => updateClient(index, "attack_strength", Number(event.target.value))}
                    />
                  </label>
                </div>
              ))}
            </div>
          </div>

          <div id="live-run" className="card timeline-card">
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
              {Object.values(processLogs).map((process) => (
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

        <section className="content-grid section-block">
          <div id="comparisons" className="card chart-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Round metrics</span>
                <h3>{chartConfig[chartMetric].title}</h3>
              </div>
              <div className="chart-toolbar">
                <div className="segmented-control compact">
                  <button
                    type="button"
                    className={chartMetric === "accuracy" ? "segment active" : "segment"}
                    onClick={() => setChartMetric("accuracy")}
                  >
                    Accuracy
                  </button>
                  <button
                    type="button"
                    className={chartMetric === "recall" ? "segment active" : "segment"}
                    onClick={() => setChartMetric("recall")}
                  >
                    Recall
                  </button>
                  <button
                    type="button"
                    className={chartMetric === "f1_score" ? "segment active" : "segment"}
                    onClick={() => setChartMetric("f1_score")}
                  >
                    F1
                  </button>
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
            </div>
            <div className="chart-wrap">
              {chartData.length ? (
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="#d9e3ea" strokeDasharray="4 4" />
                    <XAxis dataKey="round" stroke="#5f7484" allowDecimals={false} />
                    <YAxis stroke="#5f7484" domain={[0, 1]} />
                    <Tooltip />
                    <Legend />
                    <ReferenceLine y={0.9} stroke="#d8e2e8" strokeDasharray="3 3" />
                    <Line
                      type="monotone"
                      dataKey={chartConfig[chartMetric].localKey}
                      stroke="#0d9488"
                      strokeWidth={3}
                      name={chartConfig[chartMetric].localLabel}
                      dot={{ r: 4 }}
                    />
                    <Line
                      type="monotone"
                      dataKey={chartConfig[chartMetric].globalKey}
                      stroke="#1b5e74"
                      strokeWidth={3}
                      name={chartConfig[chartMetric].globalLabel}
                      dot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state">No round data yet for the selected run.</div>
              )}
            </div>
            <div className="chart-caption">
              The green line is the hospital-local validation score before aggregation. The blue line is the global model
              score after both clients send updates and the server aggregates them. This chart now compares local test
              versus global test on the same split.
            </div>
          </div>

          <div id="artifacts" className="card summary-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Global model</span>
                <h3>Final aggregated outcome</h3>
              </div>
            </div>
            <div className="metric-pair-grid">
              <div>
                <span>Strategy</span>
                <strong>{globalSummary.metadata.aggregation_strategy || "--"}</strong>
              </div>
              <div>
                <span>Rounds</span>
                <strong>{globalSummary.metadata.num_rounds || "--"}</strong>
              </div>
              <div>
                <span>Final global accuracy</span>
                <strong>{formatMetric(globalSummary.finalEvaluate?.accuracy)}</strong>
              </div>
              <div>
                <span>Final global recall</span>
                <strong>{formatMetric(globalSummary.finalEvaluate?.recall)}</strong>
              </div>
              <div>
                <span>Final global F1</span>
                <strong>{formatMetric(globalSummary.finalEvaluate?.f1_score)}</strong>
              </div>
              <div>
                <span>Global loss</span>
                <strong>{formatMetric(globalSummary.finalEvaluate?.loss)}</strong>
              </div>
            </div>
            <div className="global-summary-note">
              <p>
                Local fit metrics are the hospitals&apos; scores before aggregation. Global metrics are the server-aggregated
                model after both clients send updates back.
              </p>
            </div>
          </div>
        </section>

        <section id="clients" className="clients-board section-block">
          {clientCards.length ? (
            clientCards.map((client) => (
              <div key={client.name} className="card client-detail-card">
                <div className="section-head">
                  <div>
                    <span className="eyebrow">Hospital view</span>
                    <h3>{client.name}</h3>
                  </div>
                  <span className={badgeClass(client.summary?.metadata?.client_behavior)}>
                    {client.summary?.metadata?.client_behavior || "unknown"}
                  </span>
                </div>

                <div className="client-comparison-grid">
                  <div className="metric-panel">
                    <h4>Local model before aggregation</h4>
                    <div className="mini-metrics">
                      <div><span>Train acc</span><strong>{formatMetric(client.latestFit?.train_accuracy)}</strong></div>
                      <div><span>Train recall</span><strong>{formatMetric(client.latestFit?.train_recall)}</strong></div>
                      <div><span>Train F1</span><strong>{formatMetric(client.latestFit?.train_f1_score)}</strong></div>
                      <div><span>Local test acc</span><strong>{formatMetric(client.latestFit?.local_test_accuracy ?? client.latestFit?.val_accuracy)}</strong></div>
                      <div><span>Local test recall</span><strong>{formatMetric(client.latestFit?.local_test_recall ?? client.latestFit?.val_recall)}</strong></div>
                      <div><span>Local test F1</span><strong>{formatMetric(client.latestFit?.local_test_f1_score ?? client.latestFit?.val_f1_score)}</strong></div>
                    </div>
                  </div>

                  <div className="metric-panel">
                    <h4>Global model after aggregation</h4>
                    <div className="mini-metrics">
                      <div><span>Test acc</span><strong>{formatMetric(client.latestEvaluate?.accuracy)}</strong></div>
                      <div><span>Test recall</span><strong>{formatMetric(client.latestEvaluate?.recall)}</strong></div>
                      <div><span>Test F1</span><strong>{formatMetric(client.latestEvaluate?.f1_score)}</strong></div>
                      <div><span>Precision</span><strong>{formatMetric(client.latestEvaluate?.precision)}</strong></div>
                      <div><span>Specificity</span><strong>{formatMetric(client.latestEvaluate?.specificity)}</strong></div>
                      <div><span>Loss</span><strong>{formatMetric(client.latestEvaluate?.loss)}</strong></div>
                    </div>
                  </div>
                </div>

                <div className="delta-strip">
                  <div>
                    <span>Accuracy delta</span>
                    <strong>{formatMetric((toNumber(client.latestEvaluate?.accuracy) ?? 0) - (toNumber(client.latestFit?.local_test_accuracy ?? client.latestFit?.val_accuracy) ?? 0))}</strong>
                  </div>
                  <div>
                    <span>Recall delta</span>
                    <strong>{formatMetric((toNumber(client.latestEvaluate?.recall) ?? 0) - (toNumber(client.latestFit?.local_test_recall ?? client.latestFit?.val_recall) ?? 0))}</strong>
                  </div>
                  <div>
                    <span>F1 delta</span>
                    <strong>{formatMetric((toNumber(client.latestEvaluate?.f1_score) ?? 0) - (toNumber(client.latestFit?.local_test_f1_score ?? client.latestFit?.val_f1_score) ?? 0))}</strong>
                  </div>
                  <div>
                    <span>Confusion matrix</span>
                    <strong>
                      TP {Math.round(toNumber(client.latestEvaluate?.tp) ?? 0)} | TN {Math.round(toNumber(client.latestEvaluate?.tn) ?? 0)} | FP {Math.round(toNumber(client.latestEvaluate?.fp) ?? 0)} | FN {Math.round(toNumber(client.latestEvaluate?.fn) ?? 0)}
                    </strong>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="card empty-card">No client metrics available yet for this run.</div>
          )}
        </section>

        <section id="comparisons-table" className="grid-two section-block">
          <div id="comparisons" className="card comparison-card">
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
                      <td>{formatMetric(row.final_eval_accuracy)}</td>
                      <td>{formatMetric(row.final_eval_recall)}</td>
                      <td>{formatMetric(row.final_eval_f1_score)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div id="live-logs" className="card logs-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Process logs</span>
                <h3>Readable activity feed</h3>
              </div>
            </div>
            <div className="logs-grid">
              {Object.values(processLogs).length ? (
                Object.values(processLogs).map((process) => (
                  <div key={process.name} className="log-column">
                    <div className="log-column-head">
                      <strong>{process.name}</strong>
                      <span className={badgeClass(process.state)}>{process.state}</span>
                    </div>
                    <pre>{(process.recent_logs || []).slice(-12).join("\n") || "No logs yet."}</pre>
                  </div>
                ))
              ) : (
                <div className="empty-state">No live process logs yet.</div>
              )}
            </div>
          </div>
        </section>

        <section id="settings" className="card settings-card section-block">
          <div className="section-head">
            <div>
              <span className="eyebrow">Artifacts and settings</span>
              <h3>Saved runs</h3>
            </div>
          </div>
          <div className="saved-runs-grid">
            {dashboard.available_runs.map((run) => (
              <button
                key={run.run_name}
                className={`saved-run-card ${selectedRun === run.run_name ? "selected" : ""}`}
                onClick={() => {
                  setSelectedRun(run.run_name);
                  setActiveSection("comparisons");
                }}
              >
                <strong>{run.run_name}</strong>
                <span>{run.aggregation_strategy}</span>
                <span>Accuracy {formatMetric(run.final_accuracy)}</span>
                <span>Recall {formatMetric(run.final_recall)}</span>
                <span>F1 {formatMetric(run.final_f1_score)}</span>
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
