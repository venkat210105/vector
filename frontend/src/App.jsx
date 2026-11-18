import React, { useCallback, useEffect, useMemo, useState } from "react";
import { deleteVector, getGraph, listCollections, searchTrace, upsertVector } from "./api.js";
import CollectionPicker from "./components/CollectionPicker.jsx";
import GraphSummary from "./components/GraphSummary.jsx";
import NodeInspector from "./components/NodeInspector.jsx";
import LayerBar from "./components/LayerBar.jsx";
import ReplayControls from "./components/ReplayControls.jsx";
import SearchControls from "./components/SearchControls.jsx";
import WalkStats from "./components/WalkStats.jsx";
import { DEMO, randomPoint, randomQuery, seedDemo } from "./demo.js";
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
  const [selectedNodeId, setSelectedNodeId] = useState(null); // node shown in the inspector
  const [showDeleted, setShowDeleted] = useState(true);
  const [focusWalk, setFocusWalk] = useState(false); // hide nodes the search has not touched
  const [queryMode, setQueryMode] = useState("random"); // "random" | "custom"
  const [customText, setCustomText] = useState("");
  const [busy, setBusy] = useState(false);

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
    setSelectedNodeId(null);
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

  // Reload the graph after an edit, keeping the current layer and selection.
  const refreshGraph = async () => {
    setTrace(null);
    setGraph(await getGraph(selected.name, 600));
  };

  const addPoint = () =>
    guard(async () => {
      setBusy(true);
      try {
        const { vector, metadata } = randomPoint(selected);
        const id = `u${Date.now().toString(36)}`;
        await upsertVector(selected.name, { id, vector, metadata });
        await refreshGraph();
        setSelectedNodeId(id);
      } finally {
        setBusy(false);
      }
    });

  const deletePoint = (id) =>
    guard(async () => {
      setBusy(true);
      try {
        await deleteVector(selected.name, id);
        await refreshGraph();
      } finally {
        setBusy(false);
      }
    });

  // Parse the pasted vector; null means "fine or not in custom mode".
  const customVector = useMemo(
    () => customText.split(/[\s,]+/).filter(Boolean).map(Number),
    [customText]
  );
  const customError = (() => {
    if (queryMode !== "custom" || !selected) return null;
    if (customVector.length === 0) return "Enter a vector.";
    if (customVector.some(Number.isNaN)) return "Only numbers, separated by commas or spaces.";
    if (customVector.length !== selected.dim) return `Need ${selected.dim} numbers, got ${customVector.length}.`;
    return null;
  })();

  const runSearch = () =>
    guard(async () => {
      const vector = queryMode === "custom" ? customVector : randomQuery(selected);
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

  const selectedNode = graph?.nodes.find((n) => n.id === selectedNodeId) ?? null;
  const tombstoned = graph ? graph.nodes.length - liveCount : 0;
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
          queryMode={queryMode}
          onQueryModeChange={setQueryMode}
          customText={customText}
          onCustomTextChange={setCustomText}
          customError={customError}
          dim={selected?.dim}
          onRun={runSearch}
          disabled={!hasGraph}
        />

        {trace && <ReplayControls replay={replay} />}
        {walk && <WalkStats walk={walk} liveCount={liveCount} results={trace.results} onPick={setSelectedNodeId} />}

        {hasGraph && (
          <>
            <hr />
            <button onClick={addPoint} disabled={busy}>+ Add random point</button>
            {selectedNode ? (
              <NodeInspector
                node={selectedNode}
                graph={graph}
                layer={layer}
                busy={busy}
                onDelete={deletePoint}
                onClose={() => setSelectedNodeId(null)}
              />
            ) : (
              <p className="muted">Click a node to inspect or delete it.</p>
            )}
          </>
        )}
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
              showDeleted={showDeleted}
              onShowDeletedChange={setShowDeleted}
              tombstoned={tombstoned}
              hasWalk={!!trace}
              focusWalk={focusWalk}
              onFocusWalkChange={setFocusWalk}
            />
            <GraphCanvas
              graph={graph}
              layer={layer}
              query={query}
              walk={walk}
              selectedId={selectedNodeId}
              onSelectNode={setSelectedNodeId}
              showDeleted={showDeleted}
              focusWalk={focusWalk}
            />
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
