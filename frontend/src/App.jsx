import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getGraph, listCollections, searchTrace } from "./api.js";
import { DEMO, randomQuery, seedDemo } from "./demo.js";
import GraphCanvas from "./GraphCanvas.jsx";
import { buildEvents, walkState } from "./walk.js";

export default function App() {
  const [collections, setCollections] = useState([]);
  const [selected, setSelected] = useState(null);
  const [graph, setGraph] = useState(null);
  const [error, setError] = useState(null);
  const [seeding, setSeeding] = useState(null); // progress 0..1 while seeding

  const [k, setK] = useState(5);
  const [efSearch, setEfSearch] = useState(20);
  const [layer, setLayer] = useState(0);

  const [trace, setTrace] = useState(null);
  const [events, setEvents] = useState([]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(120); // ms per step

  const refreshCollections = useCallback(async () => {
    const list = await listCollections();
    setCollections(list);
    return list;
  }, []);

  const loadGraph = useCallback(async (name) => {
    setTrace(null);
    setEvents([]);
    setPlaying(false);
    setGraph(await getGraph(name, 600));
    setLayer(0);
  }, []);

  const guard = useCallback(async (fn) => {
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    guard(async () => {
      const list = await refreshCollections();
      const first = list.find((c) => c.index_type === "hnsw");
      if (first) setSelected(first);
    });
  }, [guard, refreshCollections]);

  useEffect(() => {
    if (selected) guard(() => loadGraph(selected.name));
  }, [selected, guard, loadGraph]);

  const createDemo = () =>
    guard(async () => {
      setSeeding(0);
      try {
        await seedDemo(setSeeding);
      } finally {
        setSeeding(null);
      }
      const list = await refreshCollections();
      const demo = list.find((c) => c.name === DEMO.name);
      setSelected(demo);
      if (demo) await loadGraph(demo.name);
    });

  const runSearch = () =>
    guard(async () => {
      const vector = randomQuery(selected);
      const result = await searchTrace(selected.name, { vector, k, ef_search: efSearch });
      setTrace(result);
      setEvents(buildEvents(result));
      setStep(0);
      setPlaying(true);
    });

  // Playback timer.
  const timer = useRef(null);
  useEffect(() => {
    clearInterval(timer.current);
    if (!playing) return;
    timer.current = setInterval(() => {
      setStep((s) => {
        if (s >= events.length - 1) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, speed);
    return () => clearInterval(timer.current);
  }, [playing, speed, events]);

  const walk = useMemo(
    () => (trace && events.length ? walkState(events, step, trace.results, graph?.total_nodes ?? 0) : null),
    [trace, events, step, graph]
  );

  // While a search is replaying, follow the layer it is currently on.
  useEffect(() => {
    if (walk) setLayer(walk.layer);
  }, [walk?.layer]); // eslint-disable-line react-hooks/exhaustive-deps

  const live = graph ? graph.nodes.filter((n) => !n.deleted).length : 0;
  const pct = walk && live ? Math.round((walk.visitedCount / live) * 100) : null;
  const query = trace ? { xy: trace.query_xy } : null;
  const hnswCollections = collections.filter((c) => c.index_type === "hnsw");

  return (
    <div className="app">
      <aside className="panel">
        <h1>vector-db explorer</h1>
        <p className="muted">Watch an HNSW search walk the layered graph.</p>

        <label>Collection</label>
        <select
          value={selected?.name ?? ""}
          onChange={(e) => setSelected(hnswCollections.find((c) => c.name === e.target.value))}
        >
          {hnswCollections.length === 0 && <option value="">no HNSW collections</option>}
          {hnswCollections.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name} ({c.dim}d, {c.metric})
            </option>
          ))}
        </select>
        <button onClick={createDemo} disabled={seeding !== null}>
          {seeding !== null ? `Seeding… ${Math.round(seeding * 100)}%` : `Create demo (${DEMO.points} points)`}
        </button>

        {graph && (
          <div className="stats">
            <div><b>{live}</b> live nodes</div>
            <div><b>{graph.max_layer + 1}</b> layers</div>
            <div>entry: <b>{graph.entry_point}</b></div>
            {graph.truncated && <div className="muted">showing {graph.nodes.length} of {graph.total_nodes}</div>}
          </div>
        )}

        <hr />
        <label>k: {k}</label>
        <input type="range" min="1" max="20" value={k} onChange={(e) => setK(+e.target.value)} />
        <label>ef_search: {efSearch}</label>
        <input type="range" min="1" max="200" value={efSearch} onChange={(e) => setEfSearch(+e.target.value)} />
        <button className="primary" onClick={runSearch} disabled={!selected || !graph || graph.nodes.length === 0}>
          Run random query
        </button>

        {events.length > 0 && (
          <>
            <div className="controls">
              <button onClick={() => { setStep(0); setPlaying(true); }}>Replay</button>
              <button onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>
              <button onClick={() => { setPlaying(false); setStep((s) => Math.min(s + 1, events.length - 1)); }}>Step</button>
            </div>
            <input
              type="range"
              min="0"
              max={events.length - 1}
              value={step}
              onChange={(e) => { setPlaying(false); setStep(+e.target.value); }}
            />
            <label>speed</label>
            <input type="range" min="20" max="600" step="20" value={620 - speed} onChange={(e) => setSpeed(620 - +e.target.value)} />
          </>
        )}

        {walk && (
          <div className="stats">
            <div>
              step {step + 1} / {events.length} — layer <b>{walk.layer}</b>
            </div>
            <div>
              nodes touched: <b>{walk.visitedCount}</b> of {live} ({pct}%)
            </div>
            {walk.showResults && (
              <ol className="results">
                {trace.results.map((r) => (
                  <li key={r.id}>
                    {r.id} <span className="muted">{r.distance.toFixed(3)}</span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
        {error && <div className="error">{error}</div>}
      </aside>

      <main className="stage">
        {graph && graph.nodes.length > 0 ? (
          <>
            <div className="layer-bar">
              <span>Viewing layer</span>
              {Array.from({ length: graph.max_layer + 1 }, (_, i) => graph.max_layer - i).map((l) => (
                <button key={l} className={l === layer ? "active" : ""} onClick={() => { setPlaying(false); setLayer(l); }}>
                  {l}
                </button>
              ))}
              <span className="legend">
                <i className="dot hot" /> current node · <i className="dot good" /> result · ✕ query · ◯ entry
              </span>
            </div>
            <GraphCanvas graph={graph} layer={layer} query={query} walk={walk} />
          </>
        ) : (
          <div className="empty">
            <p>{selected ? "This collection is empty." : "No HNSW collection yet."}</p>
            <p className="muted">Click “Create demo” to generate a clustered collection and explore it.</p>
          </div>
        )}
      </main>
    </div>
  );
}
