# Benchmarks

Recall@k, latency, memory footprint, and parameter sweeps for `HNSWIndex`,
measured against `FlatIndex` (ground truth) and `faiss-cpu`'s HNSW
implementation for credibility. Reproducible via
[`benchmarks/run_benchmark.py`](../benchmarks/run_benchmark.py) — run with
`python -m benchmarks.run_benchmark` from the repo root (takes ~90s).

**The headline finding isn't flattering, and that's the point:** at the
scales tested, this project's pure-Python `HNSWIndex` is *slower in wall
clock* than the brute-force `FlatIndex` baseline it's supposed to beat.
The rest of this doc explains exactly why, with evidence that the
*algorithm* is working correctly even though the *implementation* hasn't
earned its keep yet — those are different claims, and conflating them
would be the wrong takeaway.

## Setup

3000 random 64-dim vectors (`numpy` `default_rng`, seed 42), L2 distance,
30 random queries, `k=10`, `M=16`, `ef_construction=200` unless noted.

## Recall@10 vs `ef_search`

|`ef_search`|recall@10|
|-|-|
|10|0.66|
|20|0.80|
|50|0.97|
|100|1.00|
|200|1.00|
|400|1.00|

Matches the expected shape exactly — recall climbs and saturates as the
search widens. `ef_search=100` was used as the "reasonable default" for
every other measurement below.

## Latency (N=3000, ms/query, `ef_search=100`)

|index|recall@10|p50|p95|p99|
|-|-|-|-|-|
|FlatIndex|1.00|0.579|0.727|1.214|
|HNSW (ours)|1.00|7.312|8.439|8.679|
|faiss HNSW|0.99|0.165|0.215|0.293|

`FlatIndex` is ~12x **faster** than our HNSW here, and faiss's HNSW is
~44x faster than `FlatIndex` on top of that. Re-tested at N=15,000 to see
if this was just a "too small to matter yet" artifact:

|N|FlatIndex p50|HNSW (ours) p50|
|-|-|-|
|3,000|0.579 ms|7.312 ms|
|15,000|3.974 ms|10.527 ms|

Still slower at 5x the scale. `FlatIndex` scales roughly linearly with N as
expected (0.579ms → 3.974ms, ~6.9x for a 5x larger N — consistent with
Python-loop/dispatch overhead on top of the underlying linear scan). Our
HNSW's growth is milder (7.3ms → 10.5ms, ~1.4x) — the *shape* of the
scaling is exactly what HNSW is supposed to deliver — but the gap hadn't
closed within a practical testing budget.

## Why: distance computations vs wall-clock time

The complexity argument for HNSW is about *how many vectors get compared
to the query*, not wall-clock time directly. Instrumented `HNSWIndex` to
count actual `_distance()` calls per query and compared against
`FlatIndex`'s fixed N-per-query cost:

|N|HNSW distance calls/query|as % of brute force|
|-|-|-|
|500|498|99.6%|
|1,000|874|87.4%|
|3,000|1,566|52.2%|

**The algorithm is doing exactly what it's supposed to do** — the fraction
of the dataset it needs to touch shrinks as N grows, and by N=3,000 it's
already comparing against roughly half as many vectors as brute force
would. This ratio should keep shrinking well past this test's range,
consistent with HNSW's sub-linear complexity.

So why doesn't that translate to a wall-clock win yet? `FlatIndex`'s
`l2_batch` does **one** vectorized numpy call comparing the query against
*all* N vectors in a single C-level loop — no Python interpreter overhead
per comparison, and SIMD-friendly. `HNSWIndex._search_layer` instead makes
many *individual* Python-level calls per node visited: a heap push/pop
(tuple comparison in `heapq`), a `visited` set lookup, and one `_distance`
call comparing exactly two vectors at a time. Each of those pays full
CPython interpreter overhead, repeated once per node — even though the
*total number* of comparisons is smaller, the *per-comparison cost* is far
higher than a batched numpy call's amortized cost. faiss's HNSW doesn't
have this problem because it's compiled C++, not pure Python — its
constant factor per node visited is close to `FlatIndex`'s per-element
cost, so its algorithmic advantage shows up immediately in wall-clock time
(and did, in the table above: faiss HNSW beat faiss's own would-be
`FlatIndex`-equivalent by a wide margin at the very same N=3,000 where our
HNSW lost).

**The honest conclusion:** the complexity win here is real and measurable
(distance-call counts prove it), but it's currently masked by Python's
per-operation overhead at the scales tested. A production-grade
implementation would need to close that constant-factor gap — batching
distance computations across a whole candidate frontier instead of one
node at a time, or moving the hot path to compiled code — before the
algorithmic advantage would show up as an actual latency win at these
sizes. This is a genuine limitation to be upfront about, not a result to
paper over.

## Memory footprint (N=3000, dim=64, RSS delta in an isolated subprocess)

|index|memory|
|-|-|
|raw vector data|0.7 MB (3000 x 64 x float32)|
|FlatIndex|2.3 MB|
|HNSW (ours)|1.3 MB|
|faiss HNSW|2.1 MB|

Our HNSW's lower reported RSS delta here is more likely measurement noise
at this small a scale (dict/object overhead vs a single contiguous numpy
matrix cuts both ways) than a real advantage — worth re-measuring at a
larger N before drawing a conclusion either way; not done here due to the
build-time cost of a much larger run.

## Build time (N=3000)

|index|total|per insert|
|-|-|-|
|HNSW (ours)|38.6s|12.85 ms|
|faiss HNSW|0.16s|0.053 ms|

~240x slower to build than faiss at this N. Consistent with the same
per-operation Python overhead explanation above — insert does a full
`_search_layer` (or several, across layers) per point, the same
many-small-Python-calls cost as query search pays.

## `M` sweep (N=1000, `ef_construction=200`, `ef_search=100`)

|M|recall@10|build time|
|-|-|-|
|8|1.00|6.8s|
|16|1.00|5.8s|
|32|1.00|5.1s|

All three saturate recall at this dataset size/`ef_search`, so this sweep
doesn't distinguish them well here — a harder dataset (higher dimensional,
larger N, or a lower `ef_search` that doesn't already saturate) would be
needed to see `M`'s effect on recall separate from its effect on graph
density. Noted as a limitation of this run rather than a real finding.

## Takeaways

- Recall behaves exactly as the algorithm predicts, verified against a
  real ground truth, not just "seems fine."
- The *algorithmic* complexity advantage is real and measured directly
  (distance-call counts), separate from wall-clock time.
- The *wall-clock* advantage HNSW is famous for does not show up yet at
  the scales tested here, and comparing against faiss's compiled
  implementation proves that's an implementation-overhead problem, not an
  algorithm-correctness problem — a distinction worth being able to
  articulate precisely rather than either overclaiming ("my HNSW is fast")
  or underclaiming ("HNSW doesn't work").
- Next step, if pursued: batch distance computations across a candidate
  frontier instead of one node at a time, to close the constant-factor gap
  and let the already-real algorithmic advantage actually show up in wall
  clock.
