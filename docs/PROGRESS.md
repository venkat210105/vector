# Progress Log

Running record of what was built, why, and the key design decisions behind
each milestone. Written as we go so the reasoning survives past the commit
history. For alternatives that were considered and *not* built, see
[`CONSIDERED_IDEAS.md`](CONSIDERED_IDEAS.md).

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

## Milestone 2 — HNSW index (in progress)

Goal: replace the O(n)-per-query flat scan with an approximate
nearest-neighbor index (hierarchical navigable small world graph), so search
stays fast as collection size grows — while keeping `FlatIndex` around as
the ground-truth baseline HNSW's recall is measured against.

Steps (tracked as they land):

- [x] ADR 0001 — HNSW vs IVF/PQ, why HNSW was chosen
- [x] Core data structures (Node, layered graph, layer assignment)
- [x] Insert (greedy search + neighbor-selection heuristic)
- [x] Search (layered greedy descent + `ef_search`)
- [ ] Recall tests against `FlatIndex` ground truth
- [ ] Wire into `Collection`/API as a selectable index type

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
