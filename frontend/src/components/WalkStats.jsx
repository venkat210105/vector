import React from "react";

// Shows "how much of the graph did this search touch?" -- the picture that
// backs up HNSW's whole reason to exist: far fewer comparisons than a
// brute-force scan -- and, once the replay finishes, the ranked results.
// Clicking a result selects that node in the graph.
export default function WalkStats({ walk, liveCount, results, onPick }) {
  const percent = liveCount ? Math.round((walk.visitedCount / liveCount) * 100) : 0;

  return (
    <div className="card">
      <div className="kv"><span>now on layer</span><b>{walk.layer}</b></div>
      <div className="kv"><span>nodes touched</span><b>{walk.visitedCount} / {liveCount} ({percent}%)</b></div>
      <div className="bar"><div style={{ width: `${percent}%` }} /></div>
      {walk.showResults && (
        <ol className="results">
          {results.map((r) => (
            <li key={r.id}>
              <button className="link" onClick={() => onPick(r.id)}>{r.id}</button>
              {typeof r.metadata?.cluster === "number" && <span className="badge">cluster {r.metadata.cluster}</span>}
              <span className="muted">{r.distance.toFixed(3)}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
