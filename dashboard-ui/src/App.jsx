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

const defaultConfig = {
  run_name: "fedavg_demo",
  server_address: "127.0.0.1:8080",
  dataset_case: "similar",
  client_count: 2,
  num_rounds: 5,
  local_epochs: 2,
  batch_size: 32,
  learning_rate: 0.01,
  momentum: 0.9,
};

const datasetCases = {
  different: {
    label: "Different datasets",
    body: "Each client gets a different non-IID shard, skewed by label and image appearance.",
  },
  same: {
    label: "Same dataset",
    body: "Each client gets an exact copy of the same combined dataset.",
  },
  similar: {
    label: "Similar fragments",
    body: "Each client gets a different stratified fragment from the same combined dataset.",
  },
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

function formatInteger(value) {
  const number = toNumber(value);
  return number === null ? "--" : Math.round(number).toLocaleString();
}

function formatDatasetName(value) {
  if (!value) return "--";
  return String(value).split(/[\\/]/).pop() || String(value);
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
  const processLogs = dashboard.processes || {};
  const activeRunName = selectedRun || dashboard.current_run_name || config.run_name;

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

  const chartData = useMemo(() => {
    const fitByRound = new Map();
    const evalByRound = new Map();
    (displayedRun.fit_rounds || []).forEach((row) => {
      const round = toNumber(row.round);
      if (round !== null) fitByRound.set(round, row);
    });
    (displayedRun.evaluation_rounds || []).forEach((row) => {
      const round = toNumber(row.round);
      if (round !== null) evalByRound.set(round, row);
    });

    const rounds = Array.from(new Set([...fitByRound.keys(), ...evalByRound.keys()])).sort((a, b) => a - b);
    return rounds.map((round) => ({
      round,
      train_accuracy: toNumber(fitByRound.get(round)?.train_accuracy),
      global_accuracy: toNumber(evalByRound.get(round)?.accuracy),
    }));
  }, [displayedRun]);

  const clientChartSections = useMemo(() => {
    return clientCards.map((client) => {
      const fitByRound = new Map();
      const evalByRound = new Map();
      (client.round_metrics?.round_history || []).forEach((row) => {
        const round = toNumber(row.round);
        if (round === null) return;
        if (row.phase === "fit") fitByRound.set(round, row);
        if (row.phase === "evaluate") evalByRound.set(round, row);
      });

      const rounds = Array.from(new Set([...fitByRound.keys(), ...evalByRound.keys()])).sort((a, b) => a - b);
      return {
        name: client.name,
        dataFile: formatDatasetName(client.summary?.metadata?.data_file),
        data: rounds.map((round) => ({
          round,
          local_train_accuracy: toNumber(fitByRound.get(round)?.train_accuracy),
          global_test_accuracy: toNumber(evalByRound.get(round)?.accuracy),
          local_train_loss: toNumber(fitByRound.get(round)?.train_loss),
          global_test_loss: toNumber(evalByRound.get(round)?.loss),
        })),
      };
    });
  }, [clientCards]);

  const globalChartData = useMemo(() => {
    return (displayedRun.evaluation_rounds || [])
      .map((row) => ({
        round: toNumber(row.round),
        global_accuracy: toNumber(row.accuracy),
        global_loss: toNumber(row.loss),
      }))
      .filter((row) => row.round !== null)
      .sort((a, b) => a.round - b.round);
  }, [displayedRun]);

  const liveTimeline = useMemo(() => {
    const server = processLogs.server;
    const clients = Object.values(processLogs).filter((process) => process.name !== "server");
    const fitRounds = displayedRun.fit_rounds || [];
    const evalRounds = displayedRun.evaluation_rounds || [];
    return [
      { label: "Server listening", complete: server?.state === "running" || server?.state === "finished" },
      { label: "Clients connected", complete: clients.length >= config.client_count || fitRounds.length > 0 },
      { label: "Local training", complete: fitRounds.length > 0 },
      { label: "FedAvg aggregation", complete: fitRounds.length > 0 },
      { label: "Global evaluation", complete: evalRounds.length > 0 },
    ];
  }, [config.client_count, displayedRun, processLogs]);

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

  const clearSavedRuns = async () => {
    if (!window.confirm("Clear all saved runs from the dashboard?")) return;
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/runs/clear", { method: "POST" });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Failed to clear saved runs");
      }
      setSelectedRun("");
      setSelectedRunData(null);
      await fetchDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app-shell">
      <main className="main-panel">
        <header className="topbar card">
          <div className="topbar-title">
            <div className="brand-mark">FA</div>
            <div>
              <span className="eyebrow">Standard FedAvg</span>
              <h2>{activeRunName}</h2>
              <p>Simple federated learning demo console</p>
            </div>
          </div>
          <div className="topbar-controls">
            <button className="primary-button" onClick={startRun} disabled={busy}>
              Start run
            </button>
            <button className="secondary-button" onClick={stopRun} disabled={busy}>
              Stop run
            </button>
            <button className="danger-button" onClick={clearSavedRuns} disabled={busy}>
              Clear saved runs
            </button>
          </div>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}

        <section id="setup" className="grid-two section-block">
          <div className="card config-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Run configuration</span>
                <h3>FedAvg setup</h3>
              </div>
            </div>
            <div className="form-grid">
              <label>
                Run name
                <input value={config.run_name} onChange={(event) => setConfig((current) => ({ ...current, run_name: event.target.value }))} />
              </label>
              <label>
                Local Flower address
                <input value={config.server_address} onChange={(event) => setConfig((current) => ({ ...current, server_address: event.target.value }))} />
              </label>
              <label>
                Dataset case
                <select value={config.dataset_case} onChange={(event) => setConfig((current) => ({ ...current, dataset_case: event.target.value }))}>
                  {Object.entries(datasetCases).map(([value, item]) => (
                    <option key={value} value={value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Number of clients
                <select value={config.client_count} onChange={(event) => setConfig((current) => ({ ...current, client_count: Number(event.target.value) }))}>
                  {[1, 2, 3, 4, 5].map((count) => (
                    <option key={count} value={count}>
                      {count}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Rounds
                <input type="number" value={config.num_rounds} onChange={(event) => setConfig((current) => ({ ...current, num_rounds: Number(event.target.value) }))} />
              </label>
              <label>
                Local epochs
                <input type="number" value={config.local_epochs} onChange={(event) => setConfig((current) => ({ ...current, local_epochs: Number(event.target.value) }))} />
              </label>
              <label>
                Batch size
                <input type="number" value={config.batch_size} onChange={(event) => setConfig((current) => ({ ...current, batch_size: Number(event.target.value) }))} />
              </label>
              <label>
                Learning rate
                <input type="number" step="0.001" value={config.learning_rate} onChange={(event) => setConfig((current) => ({ ...current, learning_rate: Number(event.target.value) }))} />
              </label>
            </div>

            <div className="case-preview-grid">
              {Object.entries(datasetCases).map(([value, item]) => (
                <button
                  className={`case-preview ${config.dataset_case === value ? "selected" : ""}`}
                  key={value}
                  type="button"
                  onClick={() => setConfig((current) => ({ ...current, dataset_case: value }))}
                >
                  <strong>{item.label}</strong>
                  <span>{item.body}</span>
                </button>
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
              {Object.values(processLogs).length ? (
                Object.values(processLogs).map((process) => (
                  <div key={process.name} className="process-row">
                    <div>
                      <strong>{process.name}</strong>
                      <p>{process.pid ? `PID ${process.pid}` : "Not started"}</p>
                    </div>
                    <span className={badgeClass(process.state)}>{process.state}</span>
                  </div>
                ))
              ) : (
                <div className="empty-state compact">No active local processes yet.</div>
              )}
            </div>
          </div>
        </section>

        <section id="results" className="content-grid section-block">
          <div className="card chart-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Round metrics</span>
                <h3>FedAvg accuracy by round</h3>
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
              {chartData.length ? (
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="#d9e3ea" strokeDasharray="4 4" />
                    <XAxis dataKey="round" stroke="#5f7484" allowDecimals={false} />
                    <YAxis stroke="#5f7484" domain={[0, 1]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="train_accuracy" stroke="#0d9488" strokeWidth={3} name="Aggregated train accuracy" dot={{ r: 4 }} />
                    <Line type="monotone" dataKey="global_accuracy" stroke="#1b5e74" strokeWidth={3} name="Global test accuracy" dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state">No round data yet for the selected run.</div>
              )}
            </div>
          </div>

          <div className="card summary-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Global model</span>
                <h3>Final FedAvg result</h3>
              </div>
            </div>
            <div className="metric-pair-grid">
              <div><span>Strategy</span><strong>FedAvg</strong></div>
              <div><span>Dataset case</span><strong>{globalSummary.metadata.dataset_case || displayedRun.dataset_manifest?.dataset_case || config.dataset_case}</strong></div>
              <div><span>Clients</span><strong>{globalSummary.metadata.client_count || displayedRun.dataset_manifest?.client_count || config.client_count}</strong></div>
              <div><span>Rounds</span><strong>{globalSummary.metadata.num_rounds || "--"}</strong></div>
              <div><span>Final accuracy</span><strong>{formatMetric(globalSummary.finalEvaluate?.accuracy)}</strong></div>
              <div><span>Final loss</span><strong>{formatMetric(globalSummary.finalEvaluate?.loss)}</strong></div>
              <div><span>Client updates</span><strong>{formatInteger(globalSummary.finalFit?.received_client_updates)}</strong></div>
              <div><span>Server address</span><strong>{globalSummary.metadata.server_address || config.server_address}</strong></div>
            </div>
          </div>
        </section>

        <section className="client-chart-board section-block">
          <div className="section-title-row">
            <div>
              <span className="eyebrow">Separated model views</span>
              <h3>Client models and global model over rounds</h3>
            </div>
            <p>These charts separate local client behavior from the final global model evaluation.</p>
          </div>

          <div className="client-chart-grid">
            {clientChartSections.length ? (
              clientChartSections.map((clientChart) => (
                <div className="card chart-card" key={clientChart.name}>
                  <div className="section-head">
                    <div>
                      <span className="eyebrow">Client model</span>
                      <h3>{clientChart.name}</h3>
                    </div>
                    <span className="chart-subtitle">{clientChart.dataFile}</span>
                  </div>
                  <div className="chart-wrap compact-chart">
                    {clientChart.data.length ? (
                      <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={clientChart.data}>
                          <CartesianGrid stroke="#d9e3ea" strokeDasharray="4 4" />
                          <XAxis dataKey="round" stroke="#5f7484" allowDecimals={false} />
                          <YAxis stroke="#5f7484" domain={[0, 1]} />
                          <Tooltip />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="local_train_accuracy"
                            stroke="#0d9488"
                            strokeWidth={3}
                            name="Local train accuracy"
                            dot={{ r: 4 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="global_test_accuracy"
                            stroke="#1b5e74"
                            strokeWidth={3}
                            name="Global model test accuracy on this client"
                            dot={{ r: 4 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="empty-state">No client round data yet.</div>
                    )}
                  </div>
                  <div className="chart-caption">
                    Green is this client after local training. Blue is the server&apos;s averaged global model tested on this client.
                  </div>
                </div>
              ))
            ) : (
              <div className="card empty-card">Client charts will appear after clients finish at least one round.</div>
            )}

            <div className="card chart-card global-only-chart">
              <div className="section-head">
                <div>
                  <span className="eyebrow">Global model only</span>
                  <h3>Global evaluation over rounds</h3>
                </div>
              </div>
              <div className="chart-wrap compact-chart">
                {globalChartData.length ? (
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={globalChartData}>
                      <CartesianGrid stroke="#d9e3ea" strokeDasharray="4 4" />
                      <XAxis dataKey="round" stroke="#5f7484" allowDecimals={false} />
                      <YAxis yAxisId="accuracy" stroke="#5f7484" domain={[0, 1]} />
                      <YAxis yAxisId="loss" orientation="right" stroke="#9b6b2f" />
                      <Tooltip />
                      <Legend />
                      <Line
                        yAxisId="accuracy"
                        type="monotone"
                        dataKey="global_accuracy"
                        stroke="#1b5e74"
                        strokeWidth={3}
                        name="Global test accuracy"
                        dot={{ r: 4 }}
                      />
                      <Line
                        yAxisId="loss"
                        type="monotone"
                        dataKey="global_loss"
                        stroke="#c57d19"
                        strokeWidth={3}
                        name="Global test loss"
                        dot={{ r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty-state">No global model evaluations yet.</div>
                )}
              </div>
              <div className="chart-caption">
                This chart only shows the global model after FedAvg aggregation and client-side evaluation.
              </div>
            </div>
          </div>
        </section>

        <section id="clients" className="clients-board section-block">
          {clientCards.length ? (
            clientCards.map((client) => (
              <div key={client.name} className="card client-detail-card">
                <div className="section-head">
                  <div>
                    <span className="eyebrow">Client view</span>
                    <h3>{client.name}</h3>
                  </div>
                  <span className="badge badge-honest">FedAvg client</span>
                </div>
                <div className="client-comparison-grid">
                  <div className="metric-panel">
                    <h4>Local training update</h4>
                    <div className="mini-metrics">
                      <div><span>Train acc</span><strong>{formatMetric(client.latestFit?.train_accuracy)}</strong></div>
                      <div><span>Train loss</span><strong>{formatMetric(client.latestFit?.train_loss)}</strong></div>
                      <div><span>Tensors sent</span><strong>{formatInteger(client.latestFit?.tensor_count)}</strong></div>
                      <div><span>Scalars sent</span><strong>{formatInteger(client.latestFit?.scalar_count)}</strong></div>
                      <div><span>Dataset</span><strong>{formatDatasetName(client.summary?.metadata?.data_file)}</strong></div>
                      <div><span>Server</span><strong>{client.summary?.metadata?.server_address || "--"}</strong></div>
                    </div>
                  </div>
                  <div className="metric-panel">
                    <h4>Global model evaluation</h4>
                    <div className="mini-metrics">
                      <div><span>Accuracy</span><strong>{formatMetric(client.latestEvaluate?.accuracy)}</strong></div>
                      <div><span>Loss</span><strong>{formatMetric(client.latestEvaluate?.loss)}</strong></div>
                      <div><span>Data file</span><strong>{formatDatasetName(client.summary?.metadata?.data_file)}</strong></div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="card empty-card">No client metrics available yet for this run.</div>
          )}
        </section>

        <section id="saved-runs" className="card settings-card section-block">
          <div className="section-head">
            <div>
              <span className="eyebrow">Artifacts</span>
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
                }}
              >
                <strong>{run.run_name}</strong>
                <span>FedAvg</span>
                <span>{run.dataset_case || "manual"} · {run.client_count || "--"} client(s)</span>
                <span>Accuracy {formatMetric(run.final_accuracy)}</span>
                <span>Rounds {run.num_rounds}</span>
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
