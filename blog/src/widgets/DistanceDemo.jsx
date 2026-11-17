import React, { useState } from "react";

// Two 2D vectors you can change with sliders. Shows the L2 and cosine
// distance computed live, using the same formulas as vectordb/core/distance.py.
const clamp = (v) => Math.max(-5, Math.min(5, v));

function Slider({ label, value, onChange }) {
  return (
    <label className="slider">
      <span>{label} = <b>{value.toFixed(1)}</b></span>
      <input type="range" min="-5" max="5" step="0.5" value={value} onChange={(e) => onChange(+e.target.value)} />
    </label>
  );
}

export default function DistanceDemo() {
  const [a, setA] = useState([3, 1]);
  const [b, setB] = useState([2, 3]);

  const lenA = Math.hypot(...a);
  const lenB = Math.hypot(...b);
  const l2 = Math.hypot(a[0] - b[0], a[1] - b[1]);
  const dot = a[0] * b[0] + a[1] * b[1];
  const lengthProduct = lenA * lenB || 1e-10; // same guard as EPSILON in the code
  const cosine = 1 - dot / lengthProduct;
  const angle = (Math.acos(Math.max(-1, Math.min(1, dot / lengthProduct))) * 180) / Math.PI;

  const S = 22; // pixels per unit
  const C = 130; // center of the 260x260 plot
  const pt = (v) => [C + v[0] * S, C - v[1] * S];
  const [ax, ay] = pt(a);
  const [bx, by] = pt(b);

  return (
    <div className="widget">
      <h3>Try it: two vectors</h3>
      <div className="widget-row">
        <svg viewBox="0 0 260 260" role="img" aria-label="Two vectors drawn from the origin" className="plot">
          <line x1="0" y1={C} x2="260" y2={C} className="axis" />
          <line x1={C} y1="0" x2={C} y2="260" className="axis" />
          <line x1={ax} y1={ay} x2={bx} y2={by} className="gap" />
          <line x1={C} y1={C} x2={ax} y2={ay} className="vec a" />
          <line x1={C} y1={C} x2={bx} y2={by} className="vec b" />
          <circle cx={ax} cy={ay} r="5" className="dot a" />
          <circle cx={bx} cy={by} r="5" className="dot b" />
          <text x={ax + 7} y={ay - 7} className="lbl">A</text>
          <text x={bx + 7} y={by - 7} className="lbl">B</text>
        </svg>
        <div className="controls">
          <Slider label="A x" value={a[0]} onChange={(v) => setA([clamp(v), a[1]])} />
          <Slider label="A y" value={a[1]} onChange={(v) => setA([a[0], clamp(v)])} />
          <Slider label="B x" value={b[0]} onChange={(v) => setB([clamp(v), b[1]])} />
          <Slider label="B y" value={b[1]} onChange={(v) => setB([b[0], clamp(v)])} />
        </div>
      </div>
      <table className="results-table">
        <tbody>
          <tr><td>L2 distance (the dashed line)</td><td><b>{l2.toFixed(3)}</b></td></tr>
          <tr><td>Angle between A and B</td><td><b>{angle.toFixed(1)}°</b></td></tr>
          <tr><td>Cosine distance = 1 − cos(angle)</td><td><b>{cosine.toFixed(3)}</b></td></tr>
        </tbody>
      </table>
      <p className="hint">Set B to exactly double A: L2 grows, but cosine stays at 0, because the direction is the same.</p>
    </div>
  );
}
