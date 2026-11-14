import React from "react";

// LEARN: "controlled inputs". The slider's value comes from state (`k`), and
// when the user drags it we update that state. The state is the single
// source of truth; the input just displays it. `+event.target.value`
// converts the string the browser gives us into a number (like int()).
//
// `queryMode` picks where the query vector comes from: a random one near the
// demo clusters, or numbers the user pastes in (validated against `dim`).
export default function SearchControls({
  k, onKChange, efSearch, onEfSearchChange,
  queryMode, onQueryModeChange, customText, onCustomTextChange, customError,
  dim, onRun, disabled,
}) {
  return (
    <>
      <div className="tabs" role="tablist">
        {[["random", "Random query"], ["custom", "My vector"]].map(([mode, text]) => (
          <button key={mode} role="tab" aria-selected={queryMode === mode}
            className={queryMode === mode ? "active" : ""} onClick={() => onQueryModeChange(mode)}>
            {text}
          </button>
        ))}
      </div>

      {queryMode === "custom" && (
        <>
          <textarea
            rows={3}
            value={customText}
            placeholder={`${dim ?? "N"} comma-separated numbers`}
            onChange={(e) => onCustomTextChange(e.target.value)}
          />
          {customError && <div className="error">{customError}</div>}
        </>
      )}

      <label>k (how many results): <b>{k}</b></label>
      <input type="range" min="1" max="20" value={k} onChange={(e) => onKChange(+e.target.value)} />

      <label>ef_search (search width): <b>{efSearch}</b></label>
      <input type="range" min="1" max="200" value={efSearch} onChange={(e) => onEfSearchChange(+e.target.value)} />

      <button className="primary" onClick={onRun} disabled={disabled || (queryMode === "custom" && !!customError)}>
        Run search
      </button>
    </>
  );
}
