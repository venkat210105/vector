# HNSW explorer (frontend)

A React app that draws the layered HNSW graph and replays a search step by
step. Plain JavaScript (no TypeScript), built with Vite.

```bash
npm install
npm run dev     # http://localhost:5173 (API must be running on :8000)
npm test        # unit tests for the replay logic (Node's built-in runner)
npm run build   # production build into dist/
```

## How to read this code (learning path)

Read in this order. Each file introduces one new idea and has `LEARN:`
comments where a React concept first appears.

| # | File | What it teaches |
|---|------|-----------------|
| 1 | `src/api.js` | Plain JS: `async/await`, `fetch`, errors. No React at all. |
| 2 | `src/walk.js` + `src/walk.test.js` | Turning data into data. Pure functions are easy to test. |
| 3 | `src/components/GraphSummary.jsx` | The simplest component: props in, markup out. |
| 4 | `src/components/SearchControls.jsx` | Controlled inputs: a slider whose value *is* a piece of state. |
| 5 | `src/components/CollectionPicker.jsx` | Lists with `.map()` and `key`, events, destructured props. |
| 6 | `src/hooks/useReplay.js` | `useState`, `useEffect`, `useMemo`, cleanup, and custom hooks. |
| 7 | `src/App.jsx` | How state lives in one parent and flows down to children. |
| 8 | `src/GraphCanvas.jsx` | Escaping to the browser: drawing on a `<canvas>` from an effect. |

## Five ideas that explain most of React

1. **UI is a function of state.** You never "update the label". You change
   state, and React re-runs your component to produce the new screen.
2. **Never mutate state directly.** Call the setter (`setK(5)`), don't
   assign (`k = 5`), or React won't know to redraw.
3. **Data flows down, events flow up.** Parents pass values and callback
   functions to children as props; children call the callbacks.
4. **Don't store what you can compute.** `liveCount` and `walk` in `App.jsx`
   are derived from state each render, so they can never go out of sync.
5. **Effects are for talking to the outside world** (network, timers,
   canvas). If a value can be computed during render, don't use an effect.

## Layout

```text
src/
  api.js            # backend calls (fetch wrappers)
  demo.js           # seeded clustered demo data + random queries
  walk.js           # search trace -> replayable events -> visual state
  hooks/useReplay.js
  components/       # small single-purpose UI pieces
  GraphCanvas.jsx   # the canvas drawing of one graph layer
  App.jsx           # owns state, wires everything together
```
