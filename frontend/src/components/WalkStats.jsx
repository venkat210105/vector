import React from "react";

// Shows "how much of the graph did this search touch?" -- the picture that
// backs up HNSW's whole reason to exist: far fewer comparisons than a
// brute-force scan -- and, once the replay finishes, the ranked results.
export default function WalkStats({ walk, liveCount, results }) {
  const percent = liveCount ? Math.round((walk.visitedCount / liveCount) * 100) : 0;

  return (
    <div className="stats">
      <div>
        now on layer <b>{walk.layer}</b>
      </div>
      <div>
        nodes touched: <b>{walk.visitedCount}</b> of {liveCount} ({percent}%)
      </div>
      {walk.showResults && (
        <ol className="results">
          {results.map((r) => (
            <li key={r.id}>
              {r.id} <span className="muted">{r.distance.toFixed(3)}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
