import React from "react";

// Play / pause / step / scrub / speed. All the logic lives in the
// `useReplay` hook; this component only draws buttons and forwards clicks.
export default function ReplayControls({ replay }) {
  const { step, lastStep, playing, speed } = replay;

  return (
    <>
      <div className="controls">
        <button onClick={replay.restart}>Replay</button>
        <button onClick={replay.togglePlay}>{playing ? "Pause" : "Play"}</button>
        <button onClick={replay.stepForward}>Step</button>
      </div>

      <label>
        step {step + 1} / {lastStep + 1}
      </label>
      <input type="range" min="0" max={lastStep} value={step} onChange={(e) => replay.scrubTo(+e.target.value)} />

      <label>speed</label>
      {/* The slider is "reversed" on purpose: sliding right should feel
          faster, but a faster replay means a *smaller* delay between steps. */}
      <input
        type="range"
        min="20"
        max="600"
        step="20"
        value={620 - speed}
        onChange={(e) => replay.setSpeed(620 - +e.target.value)}
      />
    </>
  );
}
