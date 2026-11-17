import React, { useEffect, useRef } from "react";

// A faint grey graph drifting behind the page: a handful of nodes that move
// slowly and link to their two nearest neighbours, like a tiny proximity
// graph. Purely decorative, so it ignores the mouse and holds still for
// visitors who ask for reduced motion.
const NODE = "rgba(150, 158, 170, 0.55)";
const EDGE = "rgba(150, 158, 170, 0.22)";

export default function Background() {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0, h = 0, nodes = [], frame = 0;

    const setup = () => {
      const dpr = window.devicePixelRatio || 1;
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.max(10, Math.min(26, Math.round((w * h) / 55000)));
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: 2 + Math.random() * 2.5,
      }));
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      for (const n of nodes) {
        if (!still) {
          n.x += n.vx;
          n.y += n.vy;
          if (n.x < -20) n.x = w + 20; else if (n.x > w + 20) n.x = -20;
          if (n.y < -20) n.y = h + 20; else if (n.y > h + 20) n.y = -20;
        }
      }
      ctx.strokeStyle = EDGE;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const a of nodes) {
        const nearest = nodes
          .filter((b) => b !== a)
          .map((b) => ({ b, d: Math.hypot(a.x - b.x, a.y - b.y) }))
          .sort((p, q) => p.d - q.d)
          .slice(0, 2);
        for (const { b, d } of nearest) {
          if (d < 260) { ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); }
        }
      }
      ctx.stroke();
      ctx.fillStyle = NODE;
      for (const n of nodes) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();
      }
      if (!still) frame = requestAnimationFrame(draw);
    };

    setup();
    draw();
    const onResize = () => { setup(); if (still) draw(); };
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(frame); window.removeEventListener("resize", onResize); };
  }, []);

  return <canvas ref={ref} className="bg-graph" aria-hidden="true" />;
}
