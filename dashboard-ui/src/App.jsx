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

const navItems = [
  { id: "setup", label: "Setup" },
  { id: "live-run", label: "Live Run" },
  { id: "clients", label: "Clients" },
  { id: "results", label: "Results" },
  { id: "saved-runs", label: "Saved Runs" },
];

const defaultConfig = {
  run_name: "fedavg_demo",
  server_address: "0.0.0.0:8080",
  num_rounds: 5,
  local_epochs: 2,
  batch_size: 32,
  learning_rate: 0.01,
  momentum: 0.9,
  clients: [
    {
      hospital_name: "Hospital_A",
      data_file: "hospital_A_data.npz",
      enabled: true,
    },
    {
      hospital_name: "Hospital_B",
      data_file: "hospital_B_data.npz",
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

function formatInteger(value) {
  const number = toNumber(value);
  return number === null ? "--" : Math.round(number).toLocaleString();
}

function badgeClass(state) {
  return `badge badge-${state || "idle"}`;
}

function getLatestByPhase(roundHistory, phase) {
  return [...(roundHistory || [])].reverse().find((entry) => entry.phase === phase) || null;
}

function metricChips(items) {
  return items.filter((item) => item.value !== undefined && item.value !== null && item.value !== "" && item.value !== "--");
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
  const [activeSection, setActiveSection] = useState("setup");

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
  const enabledClients = config.clients.filter((client) => client.enabled);

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

  const liveTimeline = useMemo(() => {
    const server = processLogs.server;
    const clients = Object.values(processLogs).filter((process) => process.name !== "server");
    const fitRounds = displayedRun.fit_rounds || [];
    const evalRounds = displayedRun.evaluation_rounds || [];
    return [
      { label: "Server listening", complete: server?.state === "running" || server?.state === "finished" },
      { label: "Clients connected", complete: clients.length >= enabledClients.length || fitRounds.length > 0 },
      { label: "Local training", complete: fitRounds.length > 0 },
      { label: "FedAvg aggregation", complete: fitRounds.length > 0 },
      { label: "Global evaluation", complete: evalRounds.length > 0 },
    ];
  }, [displayedRun, enabledClients.length, processLogs]);

  const presentationLog = useMemo(() => {
    const fitRounds = displayedRun.fit_rounds || [];
    const evalRounds = displayedRun.evaluation_rounds || [];
    const finalFit = globalSummary.finalFit || {};
    const finalEval = globalSummary.finalEvaluate || {};
    const serverAddress = globalSummary.metadata.server_address || config.server_address;
    const connectedClients = clientCards.length ? clientCards : enabledClients.map((client) => ({ name: client.hospital_name, summary: { metadata: client } }));

    const events = [
      {
        tone: "info",
        label: "Network",
        title: "FedAvg network initialized",
        body: `The server listens on ${serverAddress}. Other devices can connect their clients to this IP and port.`,
        metrics: metricChips([
          { label: "Server", value: processLogs.server?.state || "ready" },
          { label: "Clients expected", value: connectedClients.length },
          { label: "Strategy", value: "FedAvg" },
          { label: "Rounds", value: globalSummary.metadata.num_rounds || config.num_rounds },
        ]),
      },
    ];

    connectedClients.forEach((client) => {
      const latestFit = client.latestFit || {};
      const metadata = client.summary?.metadata || {};
      events.push({
        tone: latestFit.train_accuracy ? "success" : "pending",
        label: "Client update",
        title: `${client.name || metadata.hospital_name} trains locally and sends parameters`,
        body: latestFit.train_accuracy
          ? `${client.name || metadata.hospital_name} trained on its local shard and sent model weights to the server for FedAvg.`
          : `${client.name || metadata.hospital_name} is configured with ${metadata.data_file || client.data_file || "a local dataset shard"}. Waiting for its first update.`,
        metrics: metricChips([
          { label: "Data file", value: metadata.data_file || client.data_file },
          { label: "Train accuracy", value: formatMetric(latestFit.train_accuracy) },
          { label: "Train loss", value: formatMetric(latestFit.train_loss) },
          { label: "Tensors sent", value: formatInteger(latestFit.tensor_count) },
          { label: "Scalars sent", value: formatInteger(latestFit.scalar_count) },
        ]),
      });
    });

    if (fitRounds.length) {
      events.push({
        tone: "info",
        label: "Server",
        title: "Server receives client updates and applies FedAvg",
        body: "The server computes a weighted average of the client model parameters, using each client's number of training examples.",
        metrics: metricChips([
          { label: "Latest round", value: finalFit.round },
          { label: "Client updates", value: finalFit.received_client_updates },
          { label: "Aggregated train acc", value: formatMetric(finalFit.train_accuracy) },
          { label: "Aggregation", value: "FedAvg" },
        ]),
      });
    }

    if (evalRounds.length) {
      events.push({
        tone: "success",
        label: "Global model",
        title: "Final global model is evaluated by clients",
        body: "After FedAvg, the server sends the global model back to the clients and aggregates their evaluation metrics.",
        metrics: metricChips([
          { label: "Final accuracy", value: formatMetric(finalEval.accuracy) },
          { label: "Final loss", value: formatMetric(finalEval.loss) },
          { label: "Final round", value: finalEval.round },
          { label: "Evaluating clients", value: finalEval.client_count },
        ]),
      });
    } else {
      events.push({
        tone: "pending",
        label: "Global model",
        title: "Waiting for global evaluation",
        body: "Final global accuracy and loss will appear after the first federated evaluation round completes.",
        metrics: [],
      });
    }

    return events;
  }, [clientCards, config, displayedRun, enabledClients, globalSummary, processLogs]);

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
    setConfig((current) => ({
      ...current,
      clients: current.clients.map((client, clientIndex) =>
        clientIndex === index ? { ...client, [field]: value } : client
      ),
    }));
  };

  const jumpToSection = (sectionId) => {
    setActiveSection(sectionId);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">FA</div>
          <div>
            <h1>FedAvg Control Center</h1>
            <p>Simple federated learning demo console</p>
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
          <strong>{activeRunName}</strong>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar card">
          <div>
            <span className="eyebrow">Standard FedAvg</span>
            <h2>{activeRunName}</h2>
          </div>
          <div className="topbar-controls">
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

        <section id="setup" className="grid-two section-block">
          <div className="card config-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Run configuration</span>
                <h3>FedAvg setup</h3>
              </div>
            </div>
            <div className="scenario-readout">
              <strong>Basic architecture only: 1 server, enabled clients, standard FedAvg.</strong>
              <span>Use your machine IP in `SERVER_ADDRESS` when clients run from other devices.</span>
            </div>
            <div className="form-grid">
              <label>
                Run name
                <input value={config.run_name} onChange={(event) => setConfig((current) => ({ ...current, run_name: event.target.value }))} />
              </label>
              <label>
                Server address
                <input value={config.server_address} onChange={(event) => setConfig((current) => ({ ...current, server_address: event.target.value }))} />
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
                    Dataset shard
                    <input value={client.data_file} onChange={(event) => updateClient(index, "data_file", event.target.value)} />
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
              <div><span>Rounds</span><strong>{globalSummary.metadata.num_rounds || "--"}</strong></div>
              <div><span>Final accuracy</span><strong>{formatMetric(globalSummary.finalEvaluate?.accuracy)}</strong></div>
              <div><span>Final loss</span><strong>{formatMetric(globalSummary.finalEvaluate?.loss)}</strong></div>
              <div><span>Client updates</span><strong>{formatInteger(globalSummary.finalFit?.received_client_updates)}</strong></div>
              <div><span>Server address</span><strong>{globalSummary.metadata.server_address || config.server_address}</strong></div>
            </div>
            <div className="global-summary-note">
              <p>Clients train locally on their own dataset shards. The server never sees raw images; it only receives model parameters and applies FedAvg.</p>
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
                      <div><span>Dataset</span><strong>{client.summary?.metadata?.data_file || "--"}</strong></div>
                      <div><span>Server</span><strong>{client.summary?.metadata?.server_address || "--"}</strong></div>
                    </div>
                  </div>
                  <div className="metric-panel">
                    <h4>Global model evaluation</h4>
                    <div className="mini-metrics">
                      <div><span>Accuracy</span><strong>{formatMetric(client.latestEvaluate?.accuracy)}</strong></div>
                      <div><span>Loss</span><strong>{formatMetric(client.latestEvaluate?.loss)}</strong></div>
                      <div><span>Data file</span><strong>{client.summary?.metadata?.data_file || "--"}</strong></div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="card empty-card">No client metrics available yet for this run.</div>
          )}
        </section>

        <section className="grid-two section-block">
          <div className="card comparison-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Run comparison</span>
                <h3>Saved FedAvg outcomes</h3>
              </div>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Rounds</th>
                  <th>Accuracy</th>
                  <th>Loss</th>
                </tr>
              </thead>
              <tbody>
                {dashboard.comparisons.map((row) => (
                  <tr key={row.run_name}>
                    <td>{row.run_name}</td>
                    <td>{row.num_rounds}</td>
                    <td>{formatMetric(row.final_eval_accuracy)}</td>
                    <td>{formatMetric(row.final_eval_loss)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card logs-card presentation-log-card">
            <div className="section-head">
              <div>
                <span className="eyebrow">Presentation log</span>
                <h3>What this run is doing</h3>
              </div>
            </div>
            <div className="presentation-log">
              {presentationLog.map((event, index) => (
                <article className={`presentation-event ${event.tone}`} key={`${event.label}-${index}`}>
                  <div className="event-index">{index + 1}</div>
                  <div className="event-body">
                    <div className="event-heading">
                      <span className="event-label">{event.label}</span>
                      <h4>{event.title}</h4>
                    </div>
                    <p>{event.body}</p>
                    {event.metrics.length ? (
                      <div className="event-metrics">
                        {event.metrics.map((metric) => (
                          <div key={`${event.label}-${metric.label}`}>
                            <span>{metric.label}</span>
                            <strong>{metric.value}</strong>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>
          </div>
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
                  setActiveSection("results");
                }}
              >
                <strong>{run.run_name}</strong>
                <span>FedAvg</span>
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
