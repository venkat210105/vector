# vector-db

A vector database built from scratch in Python — a single-node system supporting vector upsert/delete/search with crash-recoverable persistence, built as a deep-dive portfolio project rather than a wrapper around an existing library.

## Why this project

Most AI portfolio projects call an LLM API and stop there. This one goes a level deeper: it implements the actual mechanics behind how a vector database works — approximate nearest-neighbor search, write-ahead logging for durability, and the tradeoffs involved in making all of that fast and correct at the same time.

## Status

Currently implemented (v1, milestone 1 — flat baseline + API + persistence):

- **Flat (brute-force) index** — exact kNN via full-matrix distance computation. Serves as the ground-truth baseline that the upcoming HNSW index's recall gets measured against, and as the always-working fallback while HNSW is under development.
- **Write-ahead log (WAL)** — every upsert/delete is appended to disk (length-prefixed msgpack, fsync'd) before being applied in memory.
- **Snapshotting** — periodic full-state snapshots let recovery skip replaying the entire WAL history; the WAL is truncated after each snapshot.
- **Crash recovery** — on startup, the latest snapshot is loaded and any WAL records written after it are replayed, so an unclean shutdown (`kill -9`) doesn't lose committed writes.
- **REST API** (FastAPI) — create collections, upsert/delete vectors, search by k-nearest-neighbor, inspect stats.

Not yet built (see Roadmap):

- HNSW approximate nearest-neighbor index (the flat index is O(n) per query — fine for correctness testing, not for scale)
- Concurrency layer for safe concurrent reads/writes under load
- Deletion tombstoning + compaction (the flat index currently deletes in place; HNSW will need tombstoning since removing a node from a proximity graph mid-flight breaks graph connectivity)
- Benchmark harness (recall@k, latency percentiles, memory footprint vs. `faiss-cpu`)

## Architecture

```
vectordb/
  core/
    point.py            # Vector + id + metadata
    distance.py          # L2 / cosine distance, batched with numpy
    flat_index.py         # Brute-force baseline index
    hnsw/                 # (planned) approximate nearest-neighbor index
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

```
POST   /collections                        # {name, dim, metric}
POST   /collections/{name}/vectors          # {id, vector, metadata}
DELETE /collections/{name}/vectors/{id}
POST   /collections/{name}/search           # {vector, k}
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

## Roadmap

- **HNSW index** — hierarchical navigable small-world graph for approximate nearest-neighbor search at scale, with layer assignment, greedy-descent search, and neighbor-selection heuristics per the original Malkov & Yashunin paper. See `docs/adr/0001-hnsw-vs-ivf.md` for why HNSW over IVF/PQ.
- **Concurrency** — single-writer/multiple-reader via a coarse readers-preferring RWLock; documented as the v1 scope with full copy-on-write/MVCC graph versioning as the explicit "if I had more time" answer.
- **Benchmarking** — recall@k against brute-force ground truth, p50/p95/p99 latency, memory footprint, parameter sweeps over `M`/`ef_construction`/`ef_search`, with a `faiss-cpu` comparison row for credibility.
- **Explicitly out of scope for v1** (and why): sharding, replication, product quantization/compression, multi-tenancy, dynamic rebalancing. Single-node correctness and rigorous benchmarking are prioritized over a half-built distributed layer — each of these gets a one-line "how I'd revisit this at scale" note rather than a partial implementation.
