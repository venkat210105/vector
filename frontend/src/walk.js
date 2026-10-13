// Turns the backend's per-layer search trace into a flat list of replayable
// events, and computes the visual state at any point in that list.

export function buildEvents(trace) {
  const events = [];
  for (const layer of trace.layers) {
    events.push({ type: "enter", layer: layer.layer, ef: layer.ef, entry: layer.entry });
    for (const step of layer.steps) {
      events.push({ type: "expand", layer: layer.layer, expanded: step.expanded, discovered: step.discovered });
    }
  }
  return events;
}

// State after replaying events[0..index] inclusive.
export function walkState(events, index, results, totalNodes) {
  const visited = new Set();
  const walkedEdges = new Map();
  let currentLayerEntry = null;
  let currentExpanded = null;
  let layer = events[0]?.layer ?? 0;

  for (let i = 0; i <= index && i < events.length; i++) {
    const e = events[i];
    layer = e.layer;
    if (e.type === "enter") {
      visited.add(e.entry);
      currentLayerEntry = e.entry;
      currentExpanded = null;
    } else {
      currentExpanded = e.expanded;
      const list = walkedEdges.get(e.layer) || [];
      for (const d of e.discovered) {
        visited.add(d);
        list.push([e.expanded, d]);
      }
      walkedEdges.set(e.layer, list);
    }
  }

  const finished = index >= events.length - 1;
  return {
    layer,
    visited,
    walkedEdges,
    currentLayerEntry,
    currentExpanded,
    results: results,
    showResults: finished,
    visitedCount: visited.size,
    totalNodes,
  };
}
