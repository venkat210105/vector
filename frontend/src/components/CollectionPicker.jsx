import React from "react";
import { DEMO } from "../demo.js";

// LEARN: a component is a function that takes one object, `props`, and
// returns JSX (HTML-looking syntax). The parent passes data *down* as props
// and passes functions down (like `onSelect`) so the child can tell the
// parent something happened. Data flows down, events flow up.
//
// `{ collections, selected, ... }` below is "destructuring": pulling named
// fields out of the props object, like `collections = props["collections"]`.
export default function CollectionPicker({ collections, selected, onSelect, onCreateDemo, seedingProgress }) {
  const isSeeding = seedingProgress !== null;

  return (
    <>
      <label>Collection</label>
      <select
        value={selected?.name ?? ""}
        // LEARN: onChange is an event handler. React calls it with an event
        // object; `event.target.value` is what the user picked.
        onChange={(event) => onSelect(collections.find((c) => c.name === event.target.value))}
      >
        {collections.length === 0 && <option value="">no HNSW collections</option>}
        {/* LEARN: to show a list, .map() each item to JSX. `key` must be a
            unique, stable id so React can tell items apart between renders. */}
        {collections.map((c) => (
          <option key={c.name} value={c.name}>
            {c.name} ({c.dim}d, {c.metric})
          </option>
        ))}
      </select>

      <button onClick={onCreateDemo} disabled={isSeeding}>
        {isSeeding ? `Seeding… ${Math.round(seedingProgress * 100)}%` : `Create demo (${DEMO.points} points)`}
      </button>
    </>
  );
}
