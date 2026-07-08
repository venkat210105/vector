# ADR 0001: HNSW over IVF/PQ for the approximate nearest-neighbor index

## Status
Accepted

## Context
`FlatIndex` (milestone 1) computes exact k-NN by scanning every stored
vector — O(n) per query. That's correct but doesn't scale: at millions of
vectors, a single query becomes a full matrix scan every time. We need an
**approximate** nearest-neighbor (ANN) index that trades a small amount of
recall for a large reduction in query latency.

The three mainstream families for this:

- **HNSW** (Hierarchical Navigable Small World) — a multi-layer proximity
  graph. Each vector is a node; search greedily walks the graph toward the
  query, starting on a sparse top layer and descending into denser layers.
- **IVF** (Inverted File Index) — partitions the vector space into clusters
  (via k-means) up front; a query only searches the `nprobe` clusters
  nearest to it, not the whole dataset.
- **PQ** (Product Quantization) — not an indexing structure by itself, but a
  *compression* technique: splits each vector into sub-vectors and
  quantizes each to a small codebook index, shrinking memory footprint at
  the cost of representing vectors approximately even before search
  happens. In practice PQ is paired with IVF (as "IVF-PQ") to both narrow
  the search space and compress what's stored in each partition.

## Decision
Build **HNSW** as the ANN index for this project.

## Rationale

- **No training phase.** IVF requires a k-means clustering pass over the
  data before it can index anything — problematic for a system designed
  around incremental single-vector upserts (see `Collection.upsert`),
  since new data can drift away from cluster centroids computed earlier,
  degrading recall until you re-cluster. HNSW has no such training step: a
  vector is inserted directly into the graph via greedy search, so it fits
  the write-as-you-go model already built.
- **Better recall/latency tradeoff at the same memory budget** for the
  workload sizes this project targets, which is the standard empirical
  result reported by the original HNSW paper (Malkov & Yashunin) and
  confirmed by widely-cited ANN benchmark suites.
- **Industry default.** Qdrant, Weaviate, Milvus, and pgvector's `hnsw`
  index type all default to HNSW for exactly this reason — it's the
  structure most worth understanding deeply for a portfolio project meant
  to demonstrate real ANN mechanics, not a niche choice.
- **PQ is compression, not indexing — explicitly deferred.** The README
  already scopes product quantization/compression as out of v1 (see
  "Explicitly out of scope for v1"). Since v1 keeps full-precision float32
  vectors everywhere (matching `Point.__post_init__`'s float32 coercion),
  adding PQ now would mean building a compression layer with no indexing
  structure benefiting from it yet. Revisit PQ only if/when memory
  footprint at scale becomes the bottleneck being optimized for.

## Consequences

- **Memory**: HNSW stores full-precision vectors *plus* multiple layers of
  neighbor-list edges per node — higher memory per vector than IVF-PQ,
  which compresses vectors and only needs one flat list of centroids in
  full precision. Accepted as a v1 tradeoff; noted in the roadmap as a
  "revisit at scale" item alongside PQ.
- **Deletion is harder than in `FlatIndex`.** Removing a node from a
  proximity graph can disconnect the neighbors that pointed to it — you
  can't just tombstone the same trivial way `FlatIndex` does, because a
  dangling graph edge (pointing at a "deleted" node) can break search
  reachability for its neighbors, not just leak one dead result. This is
  why the roadmap has a dedicated "deletion tombstoning + compaction" item
  scheduled right after the base index — full graph-aware tombstoning is
  needed before delete is considered done, and until it is, understand
  the WIP HNSW index as insert/search only.
- **Approximate by construction.** Unlike `FlatIndex`, results are not
  guaranteed to be the true top-k — recall is a tunable knob
  (`ef_search`), and the benchmark harness (still on the roadmap) measures
  recall@k against `FlatIndex` as ground truth to quantify exactly how
  approximate.
- **Randomness in construction.** Each inserted node is assigned a maximum
  layer via a randomized geometric distribution, meaning graph shape
  (and therefore exact search paths) has a random component even for the
  same input data — this is why HNSW benchmarking needs to look at recall
  distributions, not single runs.

## Alternatives considered
- **IVF alone**: simpler, but recall degrades as re-clustering is
  deferred, and centroid training is awkward for a system where writes
  arrive one vector at a time rather than in a bulk-loaded batch.
- **IVF-PQ**: best memory efficiency of the three, but couples the
  indexing decision to a compression decision we're explicitly not making
  in v1 (see README's out-of-scope list). Revisit together if v2 takes on
  larger-than-memory datasets.
- **LSH (locality-sensitive hashing)**: generally dominated by HNSW on
  recall/latency in modern ANN benchmarks for the dimensionality ranges
  typical of embedding vectors; not seriously considered.
