import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getGraph, listCollections, searchTrace } from "./api.js";
import CollectionPicker from "./components/CollectionPicker.jsx";
import GraphSummary from "./components/GraphSummary.jsx";
import LayerBar from "./components/LayerBar.jsx";
import ReplayControls from "./components/ReplayControls.jsx";
import SearchControls from "./components/SearchControls.jsx";
import WalkStats from "./components/WalkStats.jsx";
import { DEMO, randomQuery, seedDemo } from "./demo.js";
import GraphCanvas from "./GraphCanvas.jsx";
import { useReplay } from "./hooks/useReplay.js";
import { walkState } from "./walk.js";

// App is the "coordinator". It owns the data (what the server told us) and
// the user's choices (sliders, selected layer), and hands pieces of them
// down to small components. Read it top to bottom:
//   1. state          -- what the page remembers
//   2. loading data   -- talking to the backend
//   3. derived values -- things computed from state
//   4. the JSX        -- what to draw
export default function App() {
  // ---- 1. State ---------------------------------------------------------
  const [collections, setCollections] = useState([]);
  const [selected, setSelected] = useState(null); // the collection being viewed
  const [graph, setGraph] = useState(null); // nodes + edges from the backend
  const [trace, setTrace] = useState(null); // the last search's recorded walk
  const [error, setError] = useState(null);
  const [seedingProgress, setSeedingProgress] = useState(null); // null = not seeding

  const [k, setK] = useState(5);
  const [efSearch, setEfSearch] = useState(20);
  const [layer, setLayer] = useState(0); // which layer is drawn

  // All playback state (step, playing, speed, timers) lives in this hook.
  const replay = useReplay(trace);

  // ---- 2. Loading data --------------------------------------------------

  // Runs an async action and shows any error in the sidebar instead of
  // letting it vanish silently.
  const guard = useCallback(async (action) => {
    setError(null);
    try {
      return await action();
    } catch (e) {
      setError(e.message);
    }
  }, []);

  const loadGraph = useCallback(async (name) => {
    setTrace(null);
    setGraph(await getGraph(name, 600));
    setLayer(0);
  }, []);

  // LEARN: an effect with an empty dependency list `[]` runs once, when the
  // page first loads. This is the usual place to fetch initial data.
  useEffect(() => {
    guard(async () => {
      const list = await listCollections();
      setCollections(list);
      const firstHnsw = list.find((c) => c.index_type === "hnsw");
      if (firstHnsw) setSelected(firstHnsw);
    });
  }, [guard]);

  // This one re-runs whenever `selected` changes: pick a collection, load
  // its graph.
  useEffect(() => {
    if (selected) guard(() => loadGraph(selected.name));
  }, [selected, guard, loadGraph]);

  const createDemo = () =>
    guard(async () => {
      setSeedingProgress(0);
      try {
        await seedDemo(setSeedingProgress);
      } finally {
        setSeedingProgress(null);
      }
      const list = await listCollections();
      setCollections(list);
      const demo = list.find((c) => c.name === DEMO.name);
      setSelected(demo);
      if (demo) await loadGraph(demo.name);
    });

  const runSearch = () =>
    guard(async () => {
      const vector = randomQuery(selected);
      setTrace(await searchTrace(selected.name, { vector, k, ef_search: efSearch }));
    });

  // ---- 3. Derived values ------------------------------------------------
  // These are computed from state on every render. They are NOT state
  // themselves: if you can compute something from existing state, don't
  // store it separately, or the two copies can drift out of sync.

  const hnswCollections = collections.filter((c) => c.index_type === "hnsw");
  const liveCount = graph ? graph.nodes.filter((n) => !n.deleted).length : 0;

  // The visual state of the search at the current replay step.
  const walk = useMemo(
    () => (trace && replay.events.length ? walkState(replay.events, replay.step, trace.results) : null),
    [trace, replay.events, replay.step]
  );

  // While a search replays, the drawn layer follows the layer the search is
  // currently on. (The user can still click a layer button to look elsewhere.)
  const walkLayer = walk?.layer;
  useEffect(() => {
    if (walkLayer !== undefined) setLayer(walkLayer);
  }, [walkLayer]);

  const query = trace ? { xy: trace.query_xy } : null;
  const hasGraph = graph && graph.nodes.length > 0;

  // ---- 4. The JSX -------------------------------------------------------
  return (
    <div className="app">
      <aside className="panel">
        <h1>vector-db explorer</h1>
        <p className="muted">Watch an HNSW search walk the layered graph.</p>

        <CollectionPicker
          collections={hnswCollections}
          selected={selected}
          onSelect={setSelected}
          onCreateDemo={createDemo}
          seedingProgress={seedingProgress}
        />

        {graph && <GraphSummary graph={graph} liveCount={liveCount} />}

        <hr />

        <SearchControls
          k={k}
          onKChange={setK}
          efSearch={efSearch}
          onEfSearchChange={setEfSearch}
          onRun={runSearch}
          disabled={!hasGraph}
        />

        {trace && <ReplayControls replay={replay} />}
        {walk && <WalkStats walk={walk} liveCount={liveCount} results={trace.results} />}
        {error && <div className="error">{error}</div>}
      </aside>

      <main className="stage">
        {hasGraph ? (
          <>
            <LayerBar
              maxLayer={graph.max_layer}
              layer={layer}
              onSelect={(l) => {
                replay.pause(); // otherwise the replay would yank the layer right back
                setLayer(l);
              }}
            />
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
