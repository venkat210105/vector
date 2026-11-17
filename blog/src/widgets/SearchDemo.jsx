import React, { useState } from "react";

// The five steps of FlatIndex.search, on a tiny example you can change.
// Each distance stands for "the distance from the query to that row".
const ROWS = [
  { id: "a", distance: 0.4 },
  { id: "b", distance: 0.1 },
  { id: "c", distance: 0.9 },
  { id: "d", distance: 0.2 },
  { id: "e", distance: 0.6 },
];

export default function SearchDemo() {
  const [deleted, setDeleted] = useState(new Set());
  const [wanted, setWanted] = useState(2);

  const toggle = (row) => {
    const next = new Set(deleted);
    next.has(row) ? next.delete(row) : next.add(row);
    setDeleted(next);
  };

  // Step 2: deleted rows become infinitely far.
  const distances = ROWS.map((r, row) => (deleted.has(row) ? Infinity : r.distance));
  // Step 3: never ask for more results than there are live rows.
  const k = Math.min(wanted, ROWS.length - deleted.size);
  // Step 4: the k smallest distances, closest first.
  const winners =
    k <= 0
      ? []
      : distances.map((d, row) => ({ row, d })).sort((x, y) => x.d - y.d).slice(0, k);

  return (
    <div className="widget">
      <h3>Try it: search a 5-row index</h3>
      <label className="slider">
        <span>k (how many results you want) = <b>{wanted}</b></span>
        <input type="range" min="1" max="5" value={wanted} onChange={(e) => setWanted(+e.target.value)} />
      </label>

      <table className="results-table search">
        <thead>
          <tr><th>row</th><th>id</th><th>1. measured distance</th><th>deleted?</th><th>2. after deletes</th></tr>
        </thead>
        <tbody>
          {ROWS.map((r, row) => {
            const won = winners.some((w) => w.row === row);
            return (
              <tr key={r.id} className={won ? "won" : ""}>
                <td>{row}</td>
                <td>{r.id}</td>
                <td>{r.distance.toFixed(1)}</td>
                <td>
                  <input type="checkbox" aria-label={`delete row ${row}`} checked={deleted.has(row)} onChange={() => toggle(row)} />
                </td>
                <td>{distances[row] === Infinity ? "inf" : distances[row].toFixed(1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <p>
        <b>3. k</b> = min({wanted}, {ROWS.length} stored − {deleted.size} deleted) = <b>{k}</b>
      </p>
      <p>
        <b>4–5. result</b>:{" "}
        {winners.length === 0
          ? "empty list (nothing alive)"
          : winners.map((w) => `${ROWS[w.row].id} (${w.d.toFixed(1)})`).join(", ")}
      </p>
      <p className="hint">Delete rows b and d and watch the result change. Delete all five and the result is empty.</p>
    </div>
  );
}
