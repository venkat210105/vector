# Architecture

How a write travels through vector-db, and how the system rebuilds itself
after a restart. The diagrams are Mermaid, which GitHub renders directly.

The design rests on one split. Searching has to be fast, so the search
structure (the HNSW graph) lives in memory. Memory is lost when the process
dies, so every change is also recorded on disk. Everything below is the
machinery that keeps those two copies in agreement.

## 1. The layers

Each layer has one job and only talks to the layer below it. Reading the code
top to bottom follows a request down this stack.

```mermaid
flowchart TB
  C["Client: browser or curl"] -->|"HTTP + JSON"| R["routes.py / schemas.py<br/>validate request"]
  R -->|"get_or_404(name)"| G["state.py: Registry<br/>dict of open collections"]
  G --> L["collection.py: Collection<br/>one named collection"]
  L -->|"1. append, fsync"| W[("name.wal<br/>write-ahead log")]
  L -->|"2. insert"| I["Index in memory<br/>HNSWIndex or FlatIndex"]
  L -.->|"every 1000 ops"| S[("name.snapshot<br/>full state")]
```

Solid arrows run on every write. The dashed arrow runs only when the
operation counter reaches `SNAPSHOT_EVERY_N_OPS`. The two cylinders are the
only things that survive a crash.

| File | Job |
|---|---|
| `api/routes.py` | HTTP only: status codes, 400/404 handling |
| `api/schemas.py` | Pydantic models for request bodies |
| `api/state.py` | Registry: finds a collection by name |
| `collection.py` | Orders the steps: WAL first, then memory |
| `core/hnsw/index.py` | The search graph, memory only |
| `storage/*.py` | WAL, snapshot, recovery on startup |

## 2. One upsert, step by step

The order inside `Collection.upsert` is the durability guarantee: the log
record reaches disk before the in-memory graph changes. If the process dies
between those two steps, the log already has the write, and recovery replays
it.

```mermaid
sequenceDiagram
  autonumber
  participant C as Client
  participant R as routes.py
  participant G as Registry
  participant L as Collection
  participant W as WAL file
  participant I as HNSWIndex
  participant S as Snapshot
  C->>R: POST /collections/demo/vectors {id, vector, metadata}
  Note over R: Pydantic checks UpsertRequest types (422 if wrong)
  R->>G: get_or_404("demo")
  G-->>R: Collection (404 if the name is unknown)
  Note over R: len(vector) must equal index.dim (400 if not)
  R->>L: upsert(id, vector, metadata)
  L->>W: append(record)
  Note over W: length prefix + msgpack, flush, fsync
  W-->>L: durable on disk
  L->>I: insert(Point)
  Note over I: pick a layer, link neighbors (memory only)
  I-->>L: graph updated
  Note over L: count this operation
  opt every 1000th operation
    L->>S: save(index) atomically
    L->>W: truncate()
  end
  L-->>R: return
  R-->>C: 204 No Content
```

A crash can land in three places:

| Crash happens | State after restart | Outcome |
|---|---|---|
| Before the WAL write | Nothing recorded, and the client never received a `204` | The client retries. No loss, because success was never promised. |
| After the WAL write, before the index insert | The WAL has the record, memory is gone anyway | Recovery replays the record. |
| After both | Same as above | Rebuilt from the WAL or a snapshot. |

A crash in the middle of a WAL write can leave a partial record at the end of
the file. `WAL.read_from` stops when a payload is shorter than its length
prefix, so a torn final record is ignored instead of breaking recovery.

### Why snapshots exist

The WAL alone could rebuild everything by replaying it from the beginning.
That gets slower the longer the system runs, and for HNSW each replayed
`upsert` re-runs a graph insert (a search plus neighbor selection). A snapshot
stores the finished graph, so loading it skips that work. `Collection.flush`
saves the snapshot and then truncates the WAL, so the log only ever holds
writes made since the last snapshot.

The snapshot is written to `name.snapshot.tmp` and then renamed over
`name.snapshot`. The rename is atomic, so a crash during the write leaves the
previous complete snapshot untouched.

## 3. Startup after a restart

Recovery rebuilds the in-memory index from disk: load the snapshot if there is
one, then replay every log record written after it.

```mermaid
flowchart LR
  A["Server starts"] --> B["Registry scans data/<br/>for *.snapshot files"]
  B --> C{"snapshot found?"}
  C -->|"yes"| D["recover():<br/>load snapshot"]
  D --> E["replay WAL records<br/>after the snapshot"]
  E --> F["Collection ready"]
  C -->|"no, only name.wal"| X["collection is skipped<br/>WAL is never read"]
  style X stroke:#d97706,stroke-width:2px
```

## Known gaps

- **A collection with no snapshot is not reopened after an unclean exit.** A
  collection gets its first snapshot after 1000 writes or a clean shutdown.
  `Registry._load_existing` only looks for `*.snapshot` files, so a collection
  that has only a `.wal` is skipped on restart even though its writes were
  fsynced. Reproduced by creating a collection, upserting one vector, and
  starting a new `Registry` without calling `close()`. The fix needs the
  Registry to also scan `*.wal` files and a way to recover `dim`, `metric` and
  `index_type` for a collection that has no snapshot header.
- **The snapshot temp file is not fsynced before the rename.** The rename
  protects against a process crash. Surviving a power loss also needs an
  `fsync` of the temp file before `replace`.
