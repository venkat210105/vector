import React from "react";

// Buttons to choose which layer of the graph is drawn. Layer 0 is the dense
// bottom layer holding every node; higher layers are sparser "express lanes".
export default function LayerBar({ maxLayer, layer, onSelect, showDeleted, onShowDeletedChange, tombstoned, hasWalk, focusWalk, onFocusWalkChange }) {
  // Top layer first, so the buttons read like the picture: high layers up top.
  const layers = Array.from({ length: maxLayer + 1 }, (_, i) => maxLayer - i);

  return (
    <div className="layer-bar">
      <span>Layer</span>
      {layers.map((l) => (
        // LEARN: className can be computed from state -- here the selected
        // layer gets the extra "active" class, which the CSS highlights.
        <button key={l} className={l === layer ? "active" : ""} onClick={() => onSelect(l)}>
          {l}
        </button>
      ))}
      {tombstoned > 0 && (
        <label className="check">
          <input type="checkbox" checked={showDeleted} onChange={(e) => onShowDeletedChange(e.target.checked)} />
          show {tombstoned} tombstoned
        </label>
      )}
      {hasWalk && (
        <label className="check">
          <input type="checkbox" checked={focusWalk} onChange={(e) => onFocusWalkChange(e.target.checked)} />
          hide untouched nodes
        </label>
      )}
      <span className="legend">
        <i className="dot hot" /> current · <i className="dot good" /> result · <i className="dot path" /> path to selected · ✕ query · ◯ entry
      </span>
    </div>
  );
}
