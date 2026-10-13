import React from "react";

// Buttons to choose which layer of the graph is drawn. Layer 0 is the dense
// bottom layer holding every node; higher layers are sparser "express lanes".
export default function LayerBar({ maxLayer, layer, onSelect }) {
  // Top layer first, so the buttons read like the picture: high layers up top.
  const layers = Array.from({ length: maxLayer + 1 }, (_, i) => maxLayer - i);

  return (
    <div className="layer-bar">
      <span>Viewing layer</span>
      {layers.map((l) => (
        // LEARN: className can be computed from state -- here the selected
        // layer gets the extra "active" class, which the CSS highlights.
        <button key={l} className={l === layer ? "active" : ""} onClick={() => onSelect(l)}>
          {l}
        </button>
      ))}
      <span className="legend">
        <i className="dot hot" /> current node · <i className="dot good" /> result · ✕ query · ◯ entry
      </span>
    </div>
  );
}
