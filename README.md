# vector-db

A vector database built from scratch in Python — a single-node system supporting vector upsert/delete/search with crash-recoverable persistence, built as a deep-dive portfolio project rather than a wrapper around an existing library.

## Why this project

Most AI portfolio projects call an LLM API and stop there. This one goes a level deeper: it implements the actual mechanics behind how a vector database works — approximate nearest-neighbor search, write-ahead logging for durability, and the tradeoffs involved in making all of that fast and correct at the same time.

## Status

**Milestone 1 — flat baseline + API + persistence:**

- **Flat (brute-force) index** — exact kNN via full-matrix distance computation. Serves as the ground-truth baseline HNSW's recall is measured against, and as the always-working fallback for exact results.
- **Write-ahead log (WAL)** — every upsert/delete is appended to disk (length-prefixed msgpack, fsync'd) before being applied in memory.
- **Snapshotting** — periodic full-state snapshots let recovery skip replaying the entire WAL history; the WAL is truncated after each snapshot.
- **Crash recovery** — on startup, the latest snapshot is loaded and any WAL records written after it are replayed, so an unclean shutdown (`kill -9`) doesn't lose committed writes.
- **REST API** (FastAPI) — create collections, upsert/delete vectors, search by k-nearest-neighbor, inspect stats.

**Milestone 2 — HNSW approximate nearest-neighbor index:**

- Layered proximity graph with randomized layer assignment, greedy-descent insert/search, and the diversity-based neighbor-selection heuristic from the original Malkov & Yashunin paper — see `docs/adr/0001-hnsw-vs-ivf.md`.
- Selectable per-collection via the API (`index_type: "flat" | "hnsw"`), with its own snapshot persistence format sharing the same on-disk envelope as `FlatIndex`.
- Recall verified against `FlatIndex` as ground truth (`tests/test_hnsw.py`), including the recall-vs-`ef_search` tradeoff curve.
- **Deletion tombstoning** — `delete()` marks a node dead without touching its edges, so other nodes that route through it stay connected; tombstoned nodes are traversed but never returned as results. Verified with a 50%-of-graph deletion stress test (100% recall@10 on survivors, zero tombstoned ids ever returned). Edge cleanup/memory reclamation (compaction) is still deferred — see Roadmap. A real crash this surfaced (and how it was fixed) is documented in [`docs/SETBACKS.md`](docs/SETBACKS.md).

**Benchmarks** (see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)):

- Recall@10 vs `ef_search` matches theory (saturates near `ef_search≈100` on the tested dataset), verified against `faiss-cpu`'s HNSW as an external reference, not just this project's own ground truth.
- **Honest finding, reported rather than buried, then acted on**: the first benchmark run found this project's pure-Python `HNSWIndex` slower in wall-clock latency than the brute-force `FlatIndex` baseline, despite provably doing far fewer distance comparisons (confirmed by instrumentation — ~52% of brute force's comparisons at N=3,000, and falling). Diagnosed the cause (Python interpreter overhead per node visited, isolated by comparing against faiss's compiled HNSW), then closed part of the gap by batching distance computations — with a real setback along the way (naive batching made *insert* slower; fixed with a measured size threshold, not a guess). Net result: query latency ~25-40% faster, insert ~12% faster, `FlatIndex` still ahead but the gap narrowed meaningfully. Full story in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`docs/SETBACKS.md`](docs/SETBACKS.md).

Not yet built (see Roadmap):

- HNSW compaction (reclaiming tombstoned nodes' memory/edges)
- Concurrency layer for safe concurrent reads/writes under load

## Architecture

```text
vectordb/
  core/
    point.py            # Vector + id + metadata
    distance.py          # L2 / cosine distance, batched with numpy
    flat_index.py         # Brute-force baseline index
    hnsw/
      index.py             # HNSW: layered graph, insert, search
  storage/
    wal.py                # Append-only write-ahead log
    snapshot.py            # Full-state snapshot save/load
    recovery.py            # Snapshot + WAL replay on startup
  collection.py            # Ties an index to its WAL/snapshot; one per named collection
  api/
    main.py                # FastAPI app
    routes.py               # /collections, /vectors, /search endpoints
    schemas.py               # Pydantic request/response models
    state.py                  # In-process registry of open collections
```

### Persistence design

Every write goes: **WAL append (fsync'd) → apply to in-memory index**. This is write-ahead in the traditional sense — if the process crashes between those two steps, recovery replays the WAL and reconstructs the exact same state. Snapshots exist purely as a compaction mechanism so recovery doesn't have to replay an unbounded WAL from the beginning of time; after every `SNAPSHOT_EVERY_N_OPS` writes, the current index state is serialized to disk and the WAL is truncated.

The tradeoff being made explicitly: fsync-per-write is durable but caps write throughput. Batching fsyncs across a small time window or write count would trade a small durability window for higher throughput — noted here as a deliberate v1 choice, not an oversight.

## API

```text
GET    /collections                        # list all open collections
POST   /collections                        # {name, dim, metric, index_type}  -- index_type: "flat" (default) | "hnsw"
POST   /collections/{name}/vectors          # {id, vector, metadata}
DELETE /collections/{name}/vectors/{id}
POST   /collections/{name}/search           # {vector, k, ef_search}  -- ef_search optional, hnsw only
GET    /collections/{name}/stats
GET    /health
```

## Running locally

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
uvicorn vectordb.api.main:app --reload
```

See [`docs/SETUP.md`](docs/SETUP.md) for PowerShell-specific setup,
example requests for both index types, and a restart-survival check.

## Roadmap

- **HNSW compaction** — deletion tombstoning is done (edges stay intact, dead nodes are traversed-through but never returned), but tombstoned nodes still consume memory and get walked on every search forever; compaction is the still-missing pass that actually reclaims them.
- **Concurrency** — single-writer/multiple-reader via a coarse readers-preferring RWLock; documented as the v1 scope with full copy-on-write/MVCC graph versioning as the explicit "if I had more time" answer.
- **Explicitly out of scope for v1** (and why): sharding, replication, product quantization/compression, multi-tenancy, dynamic rebalancing. Single-node correctness and rigorous benchmarking are prioritized over a half-built distributed layer — each of these gets a one-line "how I'd revisit this at scale" note rather than a partial implementation.

For the detailed build log (what was built, why, and how each piece was verified), see [`docs/PROGRESS.md`](docs/PROGRESS.md). For alternatives that were considered and set aside, see [`docs/CONSIDERED_IDEAS.md`](docs/CONSIDERED_IDEAS.md). For real bugs and underperformance hit along the way — root cause and what actually fixed them — see [`docs/SETBACKS.md`](docs/SETBACKS.md).
