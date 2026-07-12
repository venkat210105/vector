# Setbacks

A running record of real problems hit during development — not ideas we
considered and passed on (see [`CONSIDERED_IDEAS.md`](CONSIDERED_IDEAS.md)
for those), but things that were built, then turned out to be broken or
to underperform, plus the root cause and what actually fixed it. Kept
separate from [`PROGRESS.md`](PROGRESS.md) because "what we built" and
"what went wrong and why" are different kinds of information — this file
is the one worth re-reading before making a similar change elsewhere in
the codebase.

---

## Setback 1: deleting the graph's only point crashed the next insert/search

**When:** Milestone 3 (HNSW deletion tombstoning).

**What broke:** `HNSWIndex.delete()` tombstones a node without touching
its edges. That's correct in general, but deleting the *only* point in
the graph left `_entry_point` referencing a dead node with zero live
neighbors. The next `insert()` or `search()` crashed with an `IndexError`
reading `_search_layer(...)[0]` off an empty result list — code
throughout the file assumed a layer search always finds *something*.

**Root cause:** Two related gaps, not one:

1. `entry_point is None` was supposed to mean "no live points exist," but
   nothing enforced that once tombstoning existed — a delete could leave
   `entry_point` non-`None` while pointing at a dead node with an
   otherwise-empty graph.
2. Deeper and easy to miss: even with live points elsewhere in the graph,
   HNSW's sparse upper layers can have *every* locally-reachable node
   tombstoned at that one layer, purely by chance, without the whole
   graph being empty. A 500-point / 50-deleted test didn't surface this;
   a 500-point / **250**-deleted stress test did.

**Fix:**

1. `delete()` now resets `_entry_point`/`_entry_layer` back to `(None,
   -1)` whenever a delete empties the graph of all live points — the same
   bootstrap state a brand-new index starts in.
2. Every place that stepped to the next layer down (`insert()`'s Phase 1
   and Phase 2, `search()`'s upper-layer descent) now keeps the previous
   `nearest` if that layer's search comes back empty, instead of assuming
   it never will — a node still holds valid adjacency lists at every layer
   up to its own height, so the old `nearest` is always safe to keep using.

**How this was actually found:** not by reasoning about the code in
isolation — by writing a deliberately aggressive test (delete half a
500-point graph) and watching it crash. The lesson generalizes: recall
and correctness tests at "10% deleted" passed cleanly and gave false
confidence; the bug only showed up once tested at "50% deleted." See
`tests/test_hnsw.py::TestHNSWDelete::test_insert_works_after_deleting_the_only_point`
and `test_deleting_bridge_nodes_preserves_connectivity_and_recall`.

---

## Setback 2: naive batching made insert *slower*, not faster

**When:** Milestone 4 follow-up (closing the wall-clock gap
[`docs/BENCHMARKS.md`](BENCHMARKS.md) found between our `HNSWIndex` and
`FlatIndex`).

**What we tried first:** the benchmark's headline finding was that our
pure-Python HNSW loses to brute-force `FlatIndex` in wall-clock latency,
even though it provably does fewer distance comparisons. The diagnosed
cause was `_search_layer` computing one neighbor's distance at a time via
individual Python-level `_distance()` calls, instead of one batched
`_distance_batch()` numpy call the way `FlatIndex` does. So the first fix
was the obvious one: always batch every neighbor-expansion step.

**What actually happened:** query latency *did* improve (~38% faster:
7.31ms → 4.55ms p50 at N=3,000) — but insert got **53% slower** (12.85ms →
19.65ms per point). Batching didn't uniformly help; it depended on
*something* about how insert and search use `_search_layer` differently.

**Root cause, found by measuring rather than guessing:** instrumented
`np.stack` calls during a 3,000-point build and found **486,767** total
calls with a **median batch size of 6** — a third of all calls batched 3
or fewer vectors. `np.stack` allocates a new array and dispatches into
numpy on every call; for a batch that small, that fixed per-call cost can
exceed what batching saves. A direct microbenchmark (scalar loop vs
batched call, at batch sizes 1 through 64) confirmed exactly where the
crossover sits: **batching only wins starting around batch size 6** — below
that, plain Python scalar calls are faster.

Insert uses a much wider `ef_construction` (200 vs search's typical
`ef_search=100`), which means it explores far more of the graph per
point — and a large share of those extra expansions turn up only a
handful of not-yet-visited neighbors (most of the graph nearby is already
visited by then), producing exactly the tiny-batch case where `np.stack`
overhead dominates. Search's narrower exploration happened to sit more
often above the crossover point, which is why it benefited from
always-batching while insert didn't.

**Fix:** a hybrid threshold (`_BATCH_THRESHOLD = 6`, set directly from the
microbenchmark, not guessed) — batch via `_distance_batch` when the
unvisited-neighbor group is at or above that size, fall back to individual
`_distance` calls otherwise. Net result at N=3,000: insert **~12% faster**
than the original unoptimized code (not just "less regressed" — genuinely
better), and query latency **~25% faster**, with zero change to which
nodes get visited or what gets returned (verified: recall numbers and
distance-computation counts are bit-for-bit identical before and after,
confirmed by re-running both the recall test suite and the
distance-instrumentation script).

![Batching before vs after](images/batching_before_after.png)

**The lesson worth remembering:** "vectorize it" is not an unconditional
win — numpy operations have real fixed overhead per call, and a
vectorized call replacing a *small* number of scalar operations can lose
to the scalar loop it was meant to replace. The fix isn't "always batch"
or "never batch," it's "measure the actual crossover point for your
specific access pattern and branch on it." This is also why the two
different call sites (insert vs search) needed to be measured together,
not just one representative case — they have different batch-size
distributions because they use different `ef` values, and a fix validated
against only one of them would have shipped a regression in the other.
