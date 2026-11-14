import React from "react";

// Details for the node the user clicked: its id, how high it reaches in the
// layer stack, its metadata, and how many graph edges touch it on the layer
// currently drawn. Also the place to delete it (which tombstones it).
export default function NodeInspector({ node, graph, layer, onDelete, onClose, busy }) {
  const degree = graph.edges.filter((e) => e.layer === layer && (e.source === node.id || e.target === node.id)).length;
  const metadata = Object.entries(node.metadata || {});

  return (
    <div className="card">
      <div className="card-head">
        <b>{node.id}</b>
        <button className="link" onClick={onClose} aria-label="Close inspector">✕</button>
      </div>
      <div className="kv"><span>top layer</span><b>{node.level}</b></div>
      <div className="kv"><span>edges on layer {layer}</span><b>{degree}</b></div>
      {node.deleted && <div className="badge bad">tombstoned</div>}
      {metadata.map(([key, value]) => (
        <div className="kv" key={key}><span>{key}</span><b>{String(value)}</b></div>
      ))}
      {!node.deleted && (
        <button className="danger" onClick={() => onDelete(node.id)} disabled={busy}>
          Delete (tombstone)
        </button>
      )}
    </div>
  );
}
