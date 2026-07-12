# Progress Log

Running record of what was built, why, and the key design decisions behind
each milestone. Written as we go so the reasoning survives past the commit
history. For alternatives that were considered and *not* built, see
[`CONSIDERED_IDEAS.md`](CONSIDERED_IDEAS.md). For real bugs and
underperformance hit along the way — root cause and what actually fixed
them — see [`SETBACKS.md`](SETBACKS.md).

---

## Milestone 1 — Flat baseline + persistence + REST API

**Goal:** a correct, always-working single-node vector store, even before any
approximate-nearest-neighbor index exists. Establishes the ground truth that
HNSW's recall gets measured against later, and the durability model every
future index has to fit into.

### `Point` (`vectordb/core/point.py`)

The stored unit: `id` + `vector` + `metadata`. `__post_init__` coerces the
vector to `np.float32` — half the memory of float64, and precision loss is
irrelevant at vector-search scale. Every vector in the system funnels through
this coercion so distance math always runs on a consistent dtype.

### Distance functions (`vectordb/core/distance.py`)

Two metrics — `l2` and `cosine` — each with a scalar form and a **batch**
form (`l2_batch`, `cosine_batch`). The batch form computes the distance from
one query to every stored vector in a single vectorized numpy call
(`np.linalg.norm(matrix - query, axis=1)`) instead of looping in Python —
roughly 50-100x faster in practice. Cosine divides out vector magnitude so
only direction matters, which matters for embeddings where magnitude often
just reflects text length, not semantic content.

### `FlatIndex` (`vectordb/core/flat_index.py`)

Brute-force ground truth: one `(n, dim)` numpy matrix (`_vectors`), plus
`_ids`/`_id_to_row` maps and a `_deleted_rows` tombstone set.

Key decision: **deletes are tombstones, not array removal.** Removing a row
from a numpy matrix is O(n) (every later row shifts) and invalidates every
other id's row index. Instead `delete()` marks the row dead;`search()` sets
its distance to `+inf` so it can never win top-k. This is also *why* HNSW
will need its own tombstoning later — same problem, harder in a graph where
neighbors point at the deleted node.

`search()` uses `np.argpartition` (O(n) unordered top-k) instead of a full
sort, since only the k winners need to be sorted afterward.

Concurrency: a single `RLock` around all mutation/read paths. This is the
"coarse lock" scope the README documents as deliberate — full lock-free or
MVCC structures are out of scope for v1.

### Persistence: WAL → Collection → Snapshot → Recovery

Designed as one story, traced through a single `upsert`:

1. **WAL** (`vectordb/storage/wal.py`) appends a length-prefixed msgpack
   record and calls `fsync` before returning. `fsync` is the line that
   actually matters — without it the OS can buffer the write in memory and a
   `kill -9` loses it even though the Python code already returned.
2. **`Collection.upsert()`** (`vectordb/collection.py`) writes to the WAL
   *first*, then applies the mutation to the in-memory `FlatIndex`. That
   ordering is the entire durability guarantee: if the process dies between
   those two lines, the WAL already has the record and recovery reconstructs
   the state that was about to exist. Reversing the order would let
   in-memory state get ahead of disk with no way back.
3. **Snapshotting**: every `SNAPSHOT_EVERY_N_OPS` (1000) ops,
   `Collection.flush()` calls `snapshot.save()`
   (`vectordb/storage/snapshot.py`), serializing the full index state to a
   temp file and atomically renaming it over the real snapshot path. The
   atomic rename prevents a half-written snapshot from corrupting state if
   the process dies mid-flush. The WAL is truncated immediately after — pure
   compaction, so recovery never has to replay from the beginning of time.
4. **Recovery** (`vectordb/storage/recovery.py`): on startup, load the latest
   snapshot (if any), then replay WAL records starting at the snapshot's
   recorded `wal_offset` — the bookmark saying "everything before this byte
   offset is already captured, don't replay it again."

Explicit tradeoff (documented in README): fsync-per-write is durable but
caps write throughput. Batching fsyncs across a time window/count would
trade a small durability window for higher throughput — a deliberate v1
choice, not an oversight.

### API layer (`vectordb/api/`)

`main.py` uses a FastAPI `lifespan` hook: `Registry` opens all existing
collections on startup, and flushes+closes every collection on shutdown —
the clean-shutdown complement to crash recovery. `routes.py` is a thin REST
wrapper (dimension checks → 400, missing collection/point → 404) that calls
straight into `Collection`. `schemas.py` is the Pydantic validation boundary
for untrusted HTTP input.

### Status at end of milestone 1

Working end-to-end: create a collection, upsert/delete/search vectors over
HTTP, kill the process, restart, and prior writes survive. No ANN index yet
— `FlatIndex` is O(n) per query, correctness-first.

---

## Milestone 2 — HNSW index

Goal: replace the O(n)-per-query flat scan with an approximate
nearest-neighbor index (hierarchical navigable small world graph), so search
stays fast as collection size grows — while keeping `FlatIndex` around as
the ground-truth baseline HNSW's recall is measured against.

Steps (tracked as they land):

- [x] ADR 0001 — HNSW vs IVF/PQ, why HNSW was chosen
- [x] Core data structures (Node, layered graph, layer assignment)
- [x] Insert (greedy search + neighbor-selection heuristic)
- [x] Search (layered greedy descent + `ef_search`)
- [x] Recall tests against `FlatIndex` ground truth
- [x] Wire into `Collection`/API as a selectable index type

### ADR 0001 — HNSW vs IVF/PQ

Decision: build HNSW, not IVF or IVF-PQ. Full writeup in
[`docs/adr/0001-hnsw-vs-ivf.md`](adr/0001-hnsw-vs-ivf.md); the short version:

- IVF needs an upfront k-means training pass, which fights the
  insert-one-vector-at-a-time model already built in `Collection.upsert`.
  HNSW has no training step — a vector inserts directly into the graph.
- PQ is a compression technique, not an indexing structure, and compression
  is explicitly out of v1 scope (README) — v1 stays full-precision float32
  everywhere. Building PQ now would compress vectors for an index that
  doesn't need it yet.
- Tradeoff accepted: HNSW's memory footprint (full vectors + multi-layer
  edge lists) is higher than IVF-PQ's, and deletion is harder than
  `FlatIndex`'s tombstone-and-ignore approach, because removing a graph
  node can disconnect neighbors that pointed to it — not just leak one
  dead result. Full delete support is deferred to its own roadmap item;
  the HNSW index being built now is insert/search only until that lands.

### Core data structures (`vectordb/core/hnsw/index.py`)

`HNSWIndex` scaffolding: no insert/search algorithm yet, just the graph
representation and layer assignment those algorithms build on next.

- **Node representation**: parallel dicts keyed by point id
  (`_vectors`, `_metadata`, `_neighbors`), matching `FlatIndex`'s style
  rather than introducing a separate `Node` class per vertex — cheaper
  (no per-vertex object allocation) and consistent with the rest of the
  codebase.
- **`_neighbors[point_id]`** is a list of adjacency lists, one per layer
  the node exists on — `_neighbors[id][0]` is that node's layer-0
  neighbors, `_neighbors[id][1]` its layer-1 neighbors (if it goes that
  high), etc.
- **`_random_level()`** assigns each new node's max layer via
  `int(-ln(random()) * (1/ln(M)))` — an exponential-decay draw, so
  ~1-1/M of nodes stop at layer 0, and each layer up shrinks by roughly a
  factor of `M` (verified empirically: 20k draws at `M=16` gave layer
  counts 18815 / 1108 / 70 / 6 / 1 — each roughly 1/16th of the last).
- **`_entry_point`/`_entry_layer`**: the fixed node every search starts
  from — always whichever node currently sits on the highest layer in the
  graph. Set once insert lands (next step); a brand-new empty index has
  no entry point yet.
- **`M` vs `M_max0`**: layer 0 gets double the max-neighbors-per-node
  budget (`2*M`) of every other layer, since it holds every node and
  benefits most from extra connectivity for recall — same choice the
  original paper makes.

### Insert (`vectordb/core/hnsw/index.py`)

Three pieces, composed:

- **`_search_layer(query, entry_points, ef, layer)`** — the core
  graph-walking primitive (greedy best-first search, not a full scan).
  Keeps two heaps: `candidates` (nodes still to expand, nearest-first) and
  `found` (best `ef` results seen so far, tracked via a negated-distance
  max-heap so the current worst is cheap to check/evict). Stops expanding
  once the nearest remaining candidate is farther than the worst result
  already found — nothing left in the frontier can improve on that.
- **`_select_neighbors(query, candidates, m)`** — the diversity heuristic
  from the paper's Algorithm 4: walk candidates nearest-first, keep one
  only if it's closer to the query than to every neighbor already picked.
  Prevents a node's edges from all clustering in one direction, which
  would hurt the graph's ability to route toward queries in an
  under-connected direction.
- **`insert(point)`** — two phases. Phase 1 descends from the current top
  layer down to `level + 1` with `ef=1` (pure greedy, one hop per layer,
  since these sparse layers only need to close distance fast). Phase 2,
  from `min(level, entry_layer)` down to 0, does the real work: search
  with the full `ef_construction` candidate width, pick neighbors via the
  diversity heuristic, wire the new node's edges, and add the reverse
  edge on each neighbor too (pruning that neighbor's list back down to
  budget with the same heuristic if it overflowed). Any layer strictly
  above the old entry layer is skipped in phase 2 on purpose — a
  brand-new top layer only has this one node on it, nothing to connect to
  yet.

**Verified, not just "doesn't crash":** inserted 2000 random 16-d vectors,
then ran `_search_layer` directly (no multi-layer descent yet — `search()`
lands next) against brute-force ground truth for 20 queries: **99%
recall@10**. Confirms the graph insert produces something genuinely
navigable, not just structurally valid.

### Search (`HNSWIndex.search`)

The public query API — reuses everything insert already built. Descends
from the entry point through the upper layers with `ef=1` (same greedy
single-hop idea as insert's Phase 1 — get roughly close fast, cheaply),
then runs the real search at layer 0 with the full `ef_search` candidate
width via `_search_layer`, and returns the top `k` as `(id, distance)`
tuples — matching `FlatIndex.search`'s return convention.

`ef_search` defaults to `max(k, ef_construction)` if the caller doesn't
override it — reusing the build-time search width as a sane query-time
default.

**Verified the actual recall/latency tradeoff, not just correctness at one
setting:** 2000 random 16-d vectors, 30 queries, recall@10 against
brute-force ground truth, varying `ef_search`:

|`ef_search`|recall@10|
|-|-|
|10|0.91|
|50|1.00|
|100|1.00|
|200|1.00|

This is exactly the shape the algorithm is supposed to produce: small
`ef_search` trades some recall for speed, larger `ef_search` converges to
(near-)exact — confirming the knob actually does what the design claims,
not just that the code runs.

### Recall test suite (`tests/test_hnsw.py`)

Turns the manual spot-checks above into a real, repeatable `pytest` suite.
Key design choice: **`FlatIndex` is used directly as the ground-truth
oracle** inside the tests (`_build_indexes` builds both an `HNSWIndex` and
a `FlatIndex` over the identical vectors), rather than hand-rolling brute
force again in the test file — this is literally the relationship the
README already describes between the two indexes, so the test asserts the
real thing rather than reimplementing a parallel truth source that could
drift out of sync with `FlatIndex` itself.

Covers:

- Recall actually improves as `ef_search` grows, and stays high (≥0.95) at
  a wide `ef_search` — not just "some number came out," a directional
  guarantee.
- The default `ef_search` (no override) gives high recall (≥0.9) out of
  the box.
- **Determinism**: same `seed` twice produces an identical graph
  (`_entry_point`, `_entry_layer`, `_neighbors` all equal) — this is what
  makes the recall thresholds above safe to assert on without flakiness;
  a test whose graph shape changed randomly between runs couldn't reliably
  assert a recall floor.
- Basic contract checks: empty index returns no results, `search` never
  returns more than `k`, results come back sorted nearest-first, wrong
  vector dimension raises `ValueError` on insert, `len()` tracks inserted
  count.

8 tests, ~20s total (building two 1000-vector indexes multiple times
across the recall tests is the dominant cost).

### Wiring HNSW into `Collection`/API

Touches `collection.py`, `storage/snapshot.py`, `storage/recovery.py`,
`api/schemas.py`, `api/routes.py`, `api/state.py`.

The largest integration step: HNSW existed only in memory until now. Making
it "selectable" honestly required persistence too — a collection that
loses everything on restart wouldn't be usable end-to-end.

**Key design decision, made explicit up front:** `HNSWIndex` has no
`delete()` (deletion tombstoning is its own roadmap item — deleting a
graph node safely requires re-linking its former neighbors, unlike
`FlatIndex`'s trivial tombstone). Rather than let that surface as a
confusing crash, `Collection.delete()` explicitly checks the index type
and raises `NotImplementedError` with a clear message *before* anything
touches the WAL — and `routes.py` translates that into an honest HTTP
`501 Not Implemented`, not a `500`.

**Persistence for HNSW** (`storage/snapshot.py`): both index types now
share one on-disk envelope — a length-prefixed msgpack header (carrying an
`index_type` marker, same pattern `dim`/`metric` already used) followed by
a numpy vector blob. `_hnsw_header`/`_load_hnsw` serialize the graph's
actual state: vectors (as a matrix, positionally aligned to an `ids`
list — same trick `FlatIndex`'s snapshot already used, since HNSW's
vectors live in a dict with no natural row order), metadata, the
`_neighbors` adjacency structure (also positionally aligned to `ids` to
avoid storing id strings twice), and `entry_point`/`entry_layer`.
`storage/recovery.py` gained an `index_type` param (only consulted when no
snapshot exists yet, same as `dim`/`metric`) and now dispatches WAL replay
to `index.insert()` for HNSW vs `index.upsert()` for `FlatIndex`, since the
two classes don't share a method name there.

**API surface**: `CreateCollectionRequest` gained `index_type: str =
"flat"` (defaults preserve all existing behavior), `StatsResponse` gained
optional `entry_point`/`max_layer` fields (populated for HNSW, `None` for
flat).

**Verified at three levels, not just "imports without error":**

1. `tests/test_collection_hnsw.py` — `Collection` directly: upsert+search
   round-trip, delete correctly raises, and critically **a full
   close-and-reopen cycle preserves an HNSW collection's data** both via
   plain WAL replay and (a separate test) via the snapshot path
   specifically, by lowering `SNAPSHOT_EVERY_N_OPS` to force a real
   snapshot write+load during the test.
2. Full existing suite (12 tests total) still green — the `snapshot.py`
   refactor didn't regress the `FlatIndex` path.
3. A live smoke test through the actual FastAPI app (`TestClient`, not
   mocked): create an `hnsw` collection, upsert two vectors, search
   returns correct nearest-first results with metadata, `stats` reports
   `entry_point`/`max_layer`, delete returns `501` with the documented
   message. This also surfaced that `fastapi`/`pydantic`/`uvicorn`/`httpx`
   were listed in `requirements.txt` but had never actually been
   installed in this project's `.venv` — meaning the API layer had
   apparently never been executed before this point. Installed from the
   existing `requirements.txt` (no new dependencies added) to actually
   exercise it.

---

## Milestone 3 — HNSW deletion tombstoning

**Goal:** make `delete()` actually work for HNSW collections (previously a
`501`), without breaking the guarantee the ADR already promised — that
removing a node can't silently disconnect other nodes that route through
it to reach the rest of the graph.

### Design (`vectordb/core/hnsw/index.py`)

`delete(point_id)` marks a point in `_deleted` but leaves its edges fully
intact — no edge rewriting, no neighbor relinking. That's the entire trick:
since the graph's topology from insert-time is never touched, every
connectivity guarantee insert already established stays valid regardless
of how many nodes get tombstoned later. The cost of that simplicity: dead
nodes keep occupying memory and get walked through on every search
forever, until a future compaction pass (still on the roadmap) actually
prunes them. Tombstoning buys *correctness*, not *cleanup* — the ADR
called this out explicitly in advance.

`_search_layer` was changed to treat tombstoned nodes as pass-through
only: they still get added to `candidates` (so search keeps exploring
through them, preserving reachability to whatever's beyond) but never to
`found` (so they can never be returned as an answer). Previously both
heaps were populated together for every discovered node; now `found` only
grows for live nodes.

### A real bug this surfaced

Full writeup with root cause: [`SETBACKS.md` § Setback
1](SETBACKS.md#setback-1-deleting-the-graphs-only-point-crashed-the-next-insertsearch).
Short version:

Deleting the graph's *only* point left `_entry_point` referencing a dead
node with zero live neighbors — the next `insert()`/`search()` crashed
with an `IndexError` reading `_search_layer(...)[0]` off an empty result.
Fixed two ways:

1. `delete()` now resets `_entry_point`/`_entry_layer` back to `(None,
   -1)` — the same bootstrap state a brand-new index starts in — whenever
   a delete empties the graph of all live points. This restores an
   invariant the rest of the code already assumed everywhere else
   (`entry_point is None` ⇔ "no live points exist").
2. A second, subtler case survived that fix: during the *upper-layer*
   greedy descent (`ef=1`), a layer's search can legitimately come back
   with zero live results even when live points exist elsewhere in the
   graph — upper layers are sparse by construction, and if enough of the
   handful of nodes living there happen to be tombstoned, the whole
   locally-reachable neighborhood at that one layer can be dead. Caught
   this with a 500-point / 50-deleted stress test, not by reasoning alone.
   Fix: if a layer's search comes back empty, keep the previous `nearest`
   rather than crash — it's still a valid node to search from at the next
   layer down (every node holds adjacency lists for every layer up to its
   own height), the search just doesn't get to improve position at that
   one layer. Applied consistently across `insert()`'s Phase 1, Phase 2,
   and `search()`'s upper-layer loop.

`insert()` also now clears `point_id` from `_deleted` on insert, so a
delete-then-reinsert of the same id correctly makes it live again with
fresh edges, rather than permanently invisible (it would otherwise still
sit in `_deleted` forever even after being "re-added").

### Verified, not just "doesn't crash"

- 500 points, delete every 10th (50 total, deliberately including
  whatever bridge nodes happen to fall on that pattern): **100% recall@10**
  against `FlatIndex`-minus-the-same-deletes, and zero deleted ids ever
  returned.
- Stress test at 50% deletion (250 of 500): still **100% recall@10**, zero
  deleted ids returned — this is what actually caught the upper-layer
  empty-result bug above; the 10%-deletion test alone wasn't aggressive
  enough to hit it.
- `tests/test_hnsw.py::TestHNSWDelete` (7 tests): unknown-id/double-delete
  return `False`, deleted points never resurface in search, `stats()`
  reports tombstoned count correctly, delete-then-reinsert makes a point
  live again, the only-point-deleted regression case, and the
  bridge-node/recall stress test.
- `Collection`/API layer: `Collection.delete()` no longer special-cases
  HNSW (the `NotImplementedError` → `501` path is gone, since it's real
  functionality now, not a documented gap) — `routes.py`'s `DELETE
  /collections/{name}/vectors/{point_id}` now returns a normal `200` for
  HNSW collections too. `tests/test_collection_hnsw.py` covers delete
  removing a point from search results, unknown-id delete, and delete
  surviving a full close-and-reopen restart cycle (tombstone state is
  already part of the existing HNSW snapshot format from Milestone 2's
  `deleted` field — no snapshot format changes needed).

---

## Milestone 4 — Benchmark harness

**Goal:** turn "I built HNSW" into a measured claim — recall@k, latency
percentiles, memory footprint, parameter sweeps, and a `faiss-cpu`
comparison row for credibility, per the roadmap.

Full report: [`docs/BENCHMARKS.md`](BENCHMARKS.md). Reproducible via
`python -m benchmarks.run_benchmark`
([`benchmarks/run_benchmark.py`](../benchmarks/run_benchmark.py)).

**The headline result is not flattering, and that's deliberate to report
honestly rather than bury:** at the scales tested (up to N=15,000), this
project's pure-Python `HNSWIndex` is *slower in wall-clock time* than the
brute-force `FlatIndex` it's supposed to beat. Chased this down properly
rather than either hiding it or accepting it at face value:

- **Confirmed the algorithm itself is correct**: instrumented actual
  distance-computation counts per query. At N=3,000, HNSW touches ~52% as
  many vectors as brute force would, and that ratio was shrinking as N
  grew across 500/1,000/3,000 — exactly the sub-linear scaling the
  algorithm promises.
- **Isolated the gap to implementation, not algorithm**, by comparing
  against `faiss-cpu`'s HNSW at the identical N=3,000: faiss's HNSW beat
  `FlatIndex` by ~3.5x in wall clock at the same point ours lost by ~12x.
  Same algorithm, compiled vs pure-Python — the difference is `FlatIndex`'s
  `l2_batch` does one vectorized numpy call over all N vectors (no
  per-comparison Python overhead), while `HNSWIndex._search_layer` pays
  full CPython interpreter cost (heap push/pop, set lookup, function call)
  *per node visited*, one at a time. Fewer total comparisons doesn't win
  if each one costs far more than a batched call's amortized cost.
- This is a real, articulable limitation, not a hidden one: closing it
  would mean batching distance computations across a candidate frontier
  instead of one node at a time, or moving the hot path to compiled code —
  a concrete "how I'd revisit this" answer rather than a vague one.

Also measured: recall@10 vs `ef_search` (matches theory, saturates at
`ef_search≈100` for this dataset), memory footprint via an isolated
subprocess RSS-delta measurement (to avoid one index's allocations
contaminating the next), build time (ours: ~240x slower than faiss at
N=3,000, consistent with the same per-node Python overhead), and a small
`M` sweep (inconclusive at this dataset size — recall saturated across all
tested `M` values, noted as a limitation of that specific run rather than
papered over as a finding).
