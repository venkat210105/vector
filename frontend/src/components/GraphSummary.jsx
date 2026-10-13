import React from "react";

// A tiny "presentational" component: no state, no effects, it just turns
// props into markup. Most of your components should look like this.
export default function GraphSummary({ graph, liveCount }) {
  return (
    <div className="stats">
      <div><b>{liveCount}</b> live nodes</div>
      <div><b>{graph.max_layer + 1}</b> layers</div>
      <div>entry: <b>{graph.entry_point}</b></div>
      {/* LEARN: `cond && <jsx/>` renders the JSX only when cond is true. */}
      {graph.truncated && (
        <div className="muted">
          showing {graph.nodes.length} of {graph.total_nodes}
        </div>
      )}
    </div>
  );
}
