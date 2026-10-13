import { useEffect, useMemo, useState } from "react";
import { buildEvents } from "../walk.js";

// LEARN: a "custom hook" is just a function whose name starts with `use` and
// which calls other hooks inside it. It lets you pull a chunk of state +
// effects out of a component and give it a name. Everything about *playing
// back a search* lives here, so App.jsx doesn't need to know how it works.
export function useReplay(trace) {
  // LEARN: useMemo remembers a computed value and only recomputes it when
  // something in the dependency list (the array at the end) changes. Here,
  // we only rebuild the event list when a new `trace` arrives, not on every
  // re-render (a re-render happens every time any state changes).
  const events = useMemo(() => (trace ? buildEvents(trace) : []), [trace]);

  // LEARN: useState gives a component memory. It returns the current value
  // and a function to change it. Calling the setter makes React re-run the
  // component so the screen shows the new value.
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(120); // milliseconds per step

  const lastStep = Math.max(events.length - 1, 0);

  // LEARN: useEffect runs code *after* the screen updates, and again
  // whenever a value in its dependency list changes. This one means: "when a
  // new trace arrives, rewind to the start and begin playing".
  useEffect(() => {
    setStep(0);
    setPlaying(trace !== null);
  }, [trace]);

  // The ticking timer. While `playing` is true, move one step forward every
  // `speed` ms. LEARN: the function returned from an effect is its cleanup;
  // React calls it before re-running the effect or when the component goes
  // away. Without clearing the interval we'd leak timers.
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setStep((s) => Math.min(s + 1, lastStep)), speed);
    return () => clearInterval(id);
  }, [playing, speed, lastStep]);

  // Stop automatically when we reach the end. (Kept separate from the timer
  // so the state updater above stays a pure function of the previous value.)
  useEffect(() => {
    if (playing && step >= lastStep) setPlaying(false);
  }, [playing, step, lastStep]);

  return {
    events,
    step,
    speed,
    playing,
    lastStep,
    setSpeed,
    togglePlay: () => setPlaying((p) => !p),
    pause: () => setPlaying(false),
    restart: () => {
      setStep(0);
      setPlaying(true);
    },
    stepForward: () => {
      setPlaying(false);
      setStep((s) => Math.min(s + 1, lastStep));
    },
    scrubTo: (n) => {
      setPlaying(false);
      setStep(n);
    },
  };
}
