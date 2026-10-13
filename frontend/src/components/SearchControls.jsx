import React from "react";

// LEARN: "controlled inputs". The slider's value comes from state (`k`), and
// when the user drags it we update that state. The state is the single
// source of truth; the input just displays it. `+event.target.value`
// converts the string the browser gives us into a number (like int()).
export default function SearchControls({ k, onKChange, efSearch, onEfSearchChange, onRun, disabled }) {
  return (
    <>
      <label>k (how many results): {k}</label>
      <input type="range" min="1" max="20" value={k} onChange={(e) => onKChange(+e.target.value)} />

      <label>ef_search (search width): {efSearch}</label>
      <input type="range" min="1" max="200" value={efSearch} onChange={(e) => onEfSearchChange(+e.target.value)} />

      <button className="primary" onClick={onRun} disabled={disabled}>
        Run random query
      </button>
    </>
  );
}
