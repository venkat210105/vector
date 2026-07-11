# Considered Ideas

Ideas floated during design discussions that were evaluated and *not* (yet)
built, with the reasoning for why. Separate from `PROGRESS.md`, which only
records what was actually implemented — this is the "what else did we think
about, and why didn't it make the cut" record, kept because that reasoning
is exactly what's useful to be able to explain later (in review, in an
interview, or to a future version of ourselves revisiting the decision).

---

<details>
<summary><b>Epsilon-relaxed pruning / backtrack buffer for <code>_search_layer</code></b></summary>

**The idea:** `_search_layer`'s current pruning rule only lets a visited
neighbor into the `candidates` queue (i.e. only lets it get expanded
further) if it's strictly closer than the current worst result in `found`:

```python
if dist_n < worst_found or len(found) < ef:
```

This means a node that's *slightly* worse than the current cutoff is
discarded permanently — including never exploring *its* neighbors — even
if the true best result was only reachable by passing through it first
(a "local dip" the greedy search can't see past). The proposal: keep a
small buffer of the most-recently-discarded "close but not quite good
enough" nodes, and if the main search exhausts its candidates without
finding a global improvement, fall back and expand through the buffered
nodes too.

**Feasibility:** Real and implementable. The simplest version isn't
actually a separate buffer with an explicit swap-back trigger — it's an
**epsilon-relaxed threshold**:

```python
if dist_n < worst_found * (1 + epsilon) or len(found) < ef:
```

This lets in anything within `epsilon`% of the current worst, giving
locally-mediocre nodes a chance to be expanded instead of being cut off
outright. This is a known category of technique in the ANN literature
(slack/epsilon-approximate search, and related to how some beam-search
variants use backtracking).

**Why discarded (for now):** Empirically, widening the pruning threshold
and simply raising `ef` both do the same fundamental thing — spend more
compute exploring more of the graph in exchange for better recall — and
tend to land on a similar recall-vs-latency curve. The epsilon-slack
version is a finer-grained knob, but the added complexity (another
hyperparameter, deciding what "worse but keep" means numerically, no
clean "leaf node" stopping point to trigger backtracking since graph
search has no leaves) buys relatively little over the existing single
`ef` knob for this project's scope. This is likely why mainstream
implementations (hnswlib, FAISS's HNSW) expose `ef` alone rather than a
separate backtracking mechanism. Would be worth revisiting if benchmarking
(roadmap item) ever showed a failure mode specifically caused by one bad
early greedy hop rather than generally-insufficient search width — that's
closer to what adaptive/learned-`ef` research directions target.

</details>
