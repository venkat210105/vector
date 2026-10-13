// Run with `npm test`. Uses Node's built-in test runner, so there is nothing
// extra to install. walk.js is plain JavaScript with no React in it, which
// is exactly what makes it easy to test.
import assert from "node:assert/strict";
import { test } from "node:test";
import { buildEvents, walkState } from "./walk.js";

// A tiny hand-made trace: layer 1 hops a -> b, then layer 0 expands b, then c.
const trace = {
  layers: [
    { layer: 1, ef: 1, entry: "a", steps: [{ expanded: "a", discovered: ["b"] }], result: ["b"] },
    {
      layer: 0,
      ef: 5,
      entry: "b",
      steps: [
        { expanded: "b", discovered: ["c", "d"] },
        { expanded: "c", discovered: ["e"] },
      ],
      result: ["c", "d", "e"],
    },
  ],
};

test("buildEvents emits an 'enter' event before each layer's expansions, top layer first", () => {
  const events = buildEvents(trace);
  assert.deepEqual(
    events.map((e) => `${e.type}:${e.layer}`),
    ["enter:1", "expand:1", "enter:0", "expand:0", "expand:0"]
  );
});

test("walkState at the first event only knows the entry node", () => {
  const state = walkState(buildEvents(trace), 0, []);
  assert.deepEqual([...state.visited], ["a"]);
  assert.equal(state.layer, 1);
  assert.equal(state.showResults, false);
});

test("walkState accumulates visited nodes and walked edges per layer", () => {
  const events = buildEvents(trace);
  const state = walkState(events, events.length - 1, []);
  assert.deepEqual([...state.visited].sort(), ["a", "b", "c", "d", "e"]);
  assert.deepEqual(state.walkedEdges.get(1), [["a", "b"]]);
  assert.deepEqual(state.walkedEdges.get(0), [["b", "c"], ["b", "d"], ["c", "e"]]);
  assert.equal(state.currentExpanded, "c");
  assert.equal(state.visitedCount, 5);
});

test("results only show once the replay reaches the final event", () => {
  const events = buildEvents(trace);
  assert.equal(walkState(events, events.length - 2, []).showResults, false);
  assert.equal(walkState(events, events.length - 1, []).showResults, true);
});

test("an empty trace produces no events", () => {
  assert.deepEqual(buildEvents({ layers: [] }), []);
});
