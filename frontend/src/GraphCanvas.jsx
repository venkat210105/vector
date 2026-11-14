import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

const CLUSTER_COLORS = ["#4f8cff", "#f0883e", "#3fb950", "#bc8cff", "#e5534b", "#39c5cf", "#d4a72c", "#f778ba"];
const PAD = 36;

const clusterColor = (node) => {
  const c = node.metadata?.cluster;
  return typeof c === "number" ? CLUSTER_COLORS[c % CLUSTER_COLORS.length] : "#8b949e";
};

// Draws one layer of the HNSW graph on a canvas. `walk` is the replay state
// derived from the search trace (visited / expanded / results), or null when
// no search has been run.
export default function GraphCanvas({ graph, layer, query, walk, selectedId, onSelectNode, showDeleted }) {
  const canvasRef = useRef(null);
  const [size, setSize] = useState({ w: 800, h: 560 });
  const [hover, setHover] = useState(null);
  // Pan/zoom: screen = base projection * k + (tx, ty). Reset with double-click.
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  const dragRef = useRef(null);

  const byId = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph]);

  // Fit all node coordinates (plus the query) into the canvas.
  const project = useMemo(() => {
    const xs = graph.nodes.map((n) => n.x).concat(query ? [query.xy[0]] : []);
    const ys = graph.nodes.map((n) => n.y).concat(query ? [query.xy[1]] : []);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const sx = (size.w - 2 * PAD) / (maxX - minX || 1);
    const sy = (size.h - 2 * PAD) / (maxY - minY || 1);
    const s = Math.min(sx, sy);
    const ox = (size.w - s * (maxX - minX)) / 2;
    const oy = (size.h - s * (maxY - minY)) / 2;
    return (x, y) => [ox + (x - minX) * s, oy + (y - minY) * s];
  }, [graph, query, size]);

  const toScreen = useCallback(
    (x, y) => {
      const [px, py] = project(x, y);
      return [px * view.k + view.tx, py * view.k + view.ty];
    },
    [project, view]
  );

  // LEARN: React attaches wheel listeners as "passive", which forbids
  // preventDefault (needed to stop the page scrolling while zooming), so we
  // attach this one by hand.
  useEffect(() => {
    const canvas = canvasRef.current;
    const onWheel = (event) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = event.clientX - rect.left, my = event.clientY - rect.top;
      const factor = Math.exp(-event.deltaY * 0.0015);
      setView((v) => {
        const k = Math.min(12, Math.max(0.5, v.k * factor));
        const r = k / v.k;
        return { k, tx: mx - (mx - v.tx) * r, ty: my - (my - v.ty) * r };
      });
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  useEffect(() => {
    const el = canvasRef.current?.parentElement;
    if (!el) return;
    const observer = new ResizeObserver(() => setSize({ w: el.clientWidth, h: Math.max(380, el.clientHeight) }));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue("--fg").trim();
    const edgeColor = css.getPropertyValue("--edge").trim();
    const accent = css.getPropertyValue("--accent").trim();
    const hot = css.getPropertyValue("--hot").trim();
    const good = css.getPropertyValue("--good").trim();
    const bad = css.getPropertyValue("--bad").trim();

    const active = graph.nodes.filter((n) => n.level >= layer && (showDeleted || !n.deleted));
    const activeIds = new Set(active.map((n) => n.id));
    const pos = (n) => toScreen(n.x, n.y);

    // Static edges for this layer.
    ctx.lineWidth = 1;
    ctx.strokeStyle = edgeColor;
    ctx.beginPath();
    for (const e of graph.edges) {
      if (e.layer !== layer) continue;
      const a = byId.get(e.source), b = byId.get(e.target);
      if (!a || !b || !activeIds.has(a.id) || !activeIds.has(b.id)) continue;
      const [x1, y1] = pos(a), [x2, y2] = pos(b);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();

    // The selected node's own edges on this layer, drawn on top.
    if (selectedId) {
      ctx.strokeStyle = fg;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (const e of graph.edges) {
        if (e.layer !== layer || (e.source !== selectedId && e.target !== selectedId)) continue;
        const a = byId.get(e.source), b = byId.get(e.target);
        if (!a || !b || !activeIds.has(a.id) || !activeIds.has(b.id)) continue;
        const [x1, y1] = pos(a), [x2, y2] = pos(b);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
      ctx.stroke();
    }

    // Edges the search actually walked on this layer: expanded -> discovered.
    if (walk) {
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (const [from, to] of walk.walkedEdges.get(layer) || []) {
        const a = byId.get(from), b = byId.get(to);
        if (!a || !b) continue;
        const [x1, y1] = pos(a), [x2, y2] = pos(b);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
      ctx.stroke();
    }

    // Nodes.
    const radius = (layer === 0 ? 3.2 : 4.5) * Math.min(2, Math.sqrt(view.k));
    for (const n of active) {
      const [x, y] = pos(n);
      const visited = walk?.visited.has(n.id);
      ctx.globalAlpha = walk && !visited ? 0.28 : 1;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      if (n.deleted) {
        ctx.strokeStyle = bad;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      } else {
        ctx.fillStyle = clusterColor(n);
        ctx.fill();
      }
      if (visited) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    const ring = (id, color, r, width) => {
      const n = byId.get(id);
      if (!n || !activeIds.has(id)) return;
      const [x, y] = pos(n);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
    };

    ring(graph.entry_point, fg, 9, 2); // global entry point
    if (selectedId) ring(selectedId, fg, 12, 2.5);
    if (hover) ring(hover.node.id, accent, 7, 2);
    if (walk?.currentLayerEntry) ring(walk.currentLayerEntry, hot, 8, 2);
    if (walk?.currentExpanded) ring(walk.currentExpanded, hot, 11, 3);
    if (walk?.showResults) {
      walk.results.forEach((r, rank) => {
        ring(r.id, good, 8, 3);
        const n = byId.get(r.id);
        if (n && activeIds.has(r.id)) {
          const [x, y] = pos(n);
          ctx.fillStyle = fg;
          ctx.font = "bold 11px system-ui, sans-serif";
          ctx.fillText(String(rank + 1), x + 10, y - 8);
        }
      });
    }

    // Query marker (an x-shaped crosshair).
    if (query) {
      const [qx, qy] = toScreen(query.xy[0], query.xy[1]);
      ctx.strokeStyle = hot;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(qx - 7, qy - 7);
      ctx.lineTo(qx + 7, qy + 7);
      ctx.moveTo(qx + 7, qy - 7);
      ctx.lineTo(qx - 7, qy + 7);
      ctx.stroke();
    }
  }, [graph, layer, query, walk, size, toScreen, byId, selectedId, showDeleted, view.k, hover]);

  const pointer = (event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  };

  const nodeAt = (mx, my) => {
    let best = null, bestDist = 10;
    for (const n of graph.nodes) {
      if (n.level < layer || (!showDeleted && n.deleted)) continue;
      const [x, y] = toScreen(n.x, n.y);
      const d = Math.hypot(x - mx, y - my);
      if (d < bestDist) { best = n; bestDist = d; }
    }
    return best;
  };

  const onDown = (event) => {
    const [mx, my] = pointer(event);
    dragRef.current = { mx, my, tx: view.tx, ty: view.ty, moved: false };
  };

  const onMove = (event) => {
    const [mx, my] = pointer(event);
    const drag = dragRef.current;
    if (drag) {
      const dx = mx - drag.mx, dy = my - drag.my;
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      if (drag.moved) {
        setView((v) => ({ ...v, tx: drag.tx + dx, ty: drag.ty + dy }));
        setHover(null);
        return;
      }
    }
    const best = nodeAt(mx, my);
    setHover(best ? { node: best, x: mx, y: my } : null);
  };

  const onUp = (event) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (drag && !drag.moved) {
      const [mx, my] = pointer(event);
      onSelectNode?.(nodeAt(mx, my)?.id ?? null);
    }
  };

  return (
    <div className="canvas-wrap">
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h, cursor: hover ? "pointer" : "grab" }}
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={() => { dragRef.current = null; setHover(null); }}
        onDoubleClick={() => setView({ k: 1, tx: 0, ty: 0 })}
      />
      <div className="canvas-hint">scroll to zoom · drag to pan · click a node · double-click resets</div>
      {hover && (
        <div className="tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          <strong>{hover.node.id}</strong>
          <div>top layer: {hover.node.level}</div>
          {hover.node.deleted && <div>tombstoned</div>}
          {Object.keys(hover.node.metadata || {}).length > 0 && <div>{JSON.stringify(hover.node.metadata)}</div>}
        </div>
      )}
    </div>
  );
}
