import heapq
import math
import random
import threading

import numpy as np

from vectordb.core.distance import DISTANCE_FNS
from vectordb.core.point import Point


class HNSWIndex:
    """Hierarchical Navigable Small World approximate nearest-neighbor index.

    Nodes live on layer 0; each layer above holds an exponentially smaller
    random subset (assigned per-node at insert time via `_random_level`),
    mirroring a skip list. Search descends from a sparse top layer down to
    layer 0, using long-range edges up top to close most of the distance in
    a few hops before refining locally.
    """

    def __init__(self, dim: int, metric: str = "l2", M: int = 16, ef_construction: int = 200, seed: int | None = None) -> None:
        if metric not in DISTANCE_FNS:
            raise ValueError(f"unknown metric: {metric}")
        self.dim = dim
        self.metric = metric
        self._distance = DISTANCE_FNS[metric]

        # Max neighbors per node per layer. Layer 0 gets 2*M since it holds
        # every node and benefits most from extra connectivity for recall.
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction

        # Normalizes the random-level draw so the expected number of nodes
        # at layer l shrinks by a factor of M per layer, per the original
        # HNSW paper (Malkov & Yashunin, section 4.1).
        self._level_mult = 1.0 / math.log(M)
        self._rng = random.Random(seed)

        self._lock = threading.RLock()

        # Parallel dicts keyed by point id, matching FlatIndex's style
        # (plain dicts/arrays) rather than a separate Node class per vertex.
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}
        self._deleted: set[str] = set()

        # neighbors[point_id] = one adjacency list per layer the node
        # exists on, e.g. [[layer-0 neighbor ids], [layer-1 neighbor ids]].
        self._neighbors: dict[str, list[list[str]]] = {}

        # Fixed starting point for every search: the node currently
        # occupying the highest layer in the graph.
        self._entry_point: str | None = None
        self._entry_layer: int = -1

    def _random_level(self) -> int:
        """Draws a max layer for a new node from an exponential-decay
        distribution: P(level >= l) shrinks geometrically with l, so most
        nodes only ever exist on layer 0 and layers above get sparser fast."""
        return int(-math.log(self._rng.random()) * self._level_mult)

    def insert(self, point: Point) -> None:
        """Inserts a point into the graph. Assumes `point.id` is either new
        or was previously deleted (in which case it's un-tombstoned and
        rewired fresh, dropping its old edges) -- re-inserting an id that's
        currently live is undefined until true upsert-with-overwrite
        semantics land."""
        if point.vector.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {point.vector.shape}")

        with self._lock:
            level = self._random_level()
            self._vectors[point.id] = point.vector
            self._metadata[point.id] = point.metadata
            self._neighbors[point.id] = [[] for _ in range(level + 1)]
            self._deleted.discard(point.id)

            if self._entry_point is None:
                self._entry_point = point.id
                self._entry_layer = level
                return

            # Phase 1: descend from the current top layer down to `level + 1`
            # with ef=1 (pure greedy, single best hop per layer). These
            # layers only exist to close most of the distance fast, so a
            # wide candidate list would be wasted effort here.
            # `result` can come back empty if everything reachable at this
            # sparse upper layer happens to be tombstoned -- `nearest`
            # itself is still a valid node to search from at the next
            # layer down regardless, so just keep it rather than crash.
            nearest = self._entry_point
            for lc in range(self._entry_layer, level, -1):
                result = self._search_layer(point.vector, [nearest], ef=1, layer=lc)
                if result:
                    nearest = result[0][1]

            # Phase 2: from min(level, entry_layer) down to 0, actually wire
            # up edges using the full ef_construction candidate width. Any
            # layer strictly above the old entry_layer is skipped here on
            # purpose -- this new node is the only occupant up there, so
            # there's nothing yet to connect it to.
            for lc in range(min(level, self._entry_layer), -1, -1):
                candidates = self._search_layer(point.vector, [nearest], ef=self.ef_construction, layer=lc)
                max_neighbors = self.M_max0 if lc == 0 else self.M
                chosen = self._select_neighbors(point.vector, candidates, max_neighbors)

                self._neighbors[point.id][lc] = list(chosen)
                for neighbor_id in chosen:
                    self._add_edge(neighbor_id, point.id, lc, max_neighbors)

                if candidates:
                    nearest = candidates[0][1]

            if level > self._entry_layer:
                self._entry_point = point.id
                self._entry_layer = level

    def delete(self, point_id: str) -> bool:
        """Tombstones a point rather than removing it from the graph: its
        edges are left fully intact so any node that routes through it to
        reach the rest of the graph stays connected. `_search_layer`
        traverses through tombstoned nodes but never returns them as a
        result. Actual edge cleanup and memory reclamation is deferred to
        a future compaction pass (roadmap) -- this alone is what makes
        delete safe, not what makes it free.
        """
        with self._lock:
            if point_id not in self._vectors or point_id in self._deleted:
                return False
            self._deleted.add(point_id)
            if len(self) == 0:
                # No live points left -- reset to the same bootstrap state
                # a brand-new empty index starts in, so the next
                # insert/search doesn't have to reason about a graph made
                # entirely of tombstones.
                self._entry_point = None
                self._entry_layer = -1
            return True

    def search(self, query: np.ndarray, k: int, ef_search: int | None = None) -> list[tuple[str, float]]:
        """Approximate k-nearest-neighbor search. Descends from the entry
        point through the upper layers greedily (ef=1, same idea as
        insert's Phase 1 -- just get roughly close fast), then does the
        real search at layer 0 with the full `ef_search` candidate width.

        `ef_search` defaults to `max(k, ef_construction)` if not given --
        reusing the build-time search width as a reasonable query-time
        default. Larger `ef_search` trades query latency for recall.
        """
        query = np.asarray(query, dtype=np.float32)
        with self._lock:
            if self._entry_point is None:
                return []
            ef = ef_search if ef_search is not None else max(k, self.ef_construction)

            nearest = self._entry_point
            for lc in range(self._entry_layer, 0, -1):
                result = self._search_layer(query, [nearest], ef=1, layer=lc)
                if result:
                    nearest = result[0][1]

            candidates = self._search_layer(query, [nearest], ef=ef, layer=0)
            return [(eid, dist) for dist, eid in sorted(candidates)[:k]]

    def _search_layer(self, query: np.ndarray, entry_points: list[str], ef: int, layer: int) -> list[tuple[float, str]]:
        """Greedy best-first search within a single layer. Explores outward
        from `entry_points` by following graph edges -- never scans every
        node -- and returns up to `ef` (distance, id) pairs, nearest first.

        Maintains two heaps: `candidates` (min-heap, nodes still to expand
        -- includes tombstoned nodes, since they must stay traversable for
        connectivity) and `found` (max-heap via negated distance, the best
        `ef` *live* results seen so far -- tombstoned nodes never enter
        this one, so they can never be returned, only passed through).
        Stops expanding once the nearest unexplored candidate is farther
        than the worst of the `ef` results already found -- nothing left
        in the frontier can possibly improve on that. While `found` is
        still empty (e.g. every node visited so far is tombstoned), that
        "worst" comparison is treated as infinitely bad, so expansion never
        stops early just because nothing live has turned up yet.
        """
        visited = set(entry_points)
        candidates = [(self._dist_to(query, ep), ep) for ep in entry_points]
        heapq.heapify(candidates)
        found = [(-dist, eid) for dist, eid in candidates if eid not in self._deleted]
        heapq.heapify(found)

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            worst_found = -found[0][0] if found else float("inf")
            if dist_c > worst_found and len(found) >= ef:
                break

            for neighbor_id in self._neighbors[c][layer]:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                dist_n = self._dist_to(query, neighbor_id)
                worst_found = -found[0][0] if found else float("inf")
                if dist_n < worst_found or len(found) < ef:
                    heapq.heappush(candidates, (dist_n, neighbor_id))
                    if neighbor_id not in self._deleted:
                        heapq.heappush(found, (-dist_n, neighbor_id))
                        if len(found) > ef:
                            heapq.heappop(found)

        return sorted((-dist, eid) for dist, eid in found)

    def _select_neighbors(self, query: np.ndarray, candidates: list[tuple[float, str]], m: int) -> list[str]:
        """Picks up to `m` ids from `candidates`, nearest-first, keeping a
        candidate only if it's closer to `query` than to every neighbor
        already selected. This is the diversity heuristic from the HNSW
        paper (Algorithm 4) -- naively taking the `m` nearest tends to
        cluster all of a node's edges in one direction, which hurts the
        graph's ability to route searches toward query points that lie in
        an under-connected direction.
        """
        selected: list[str] = []
        for dist_to_query, candidate_id in sorted(candidates):
            if len(selected) >= m:
                break
            candidate_vec = self._vectors[candidate_id]
            if all(self._distance(candidate_vec, self._vectors[s]) > dist_to_query for s in selected):
                selected.append(candidate_id)
        return selected

    def _add_edge(self, neighbor_id: str, new_id: str, layer: int, max_neighbors: int) -> None:
        """Wires the reverse edge (neighbor -> new node) and, if that pushed
        `neighbor_id` over its budget for this layer, re-prunes its edge
        list down to `max_neighbors` using the same diversity heuristic,
        viewed from `neighbor_id`'s own vantage point."""
        edges = self._neighbors[neighbor_id][layer]
        edges.append(new_id)
        if len(edges) > max_neighbors:
            neighbor_vec = self._vectors[neighbor_id]
            candidates = [(self._distance(neighbor_vec, self._vectors[eid]), eid) for eid in edges]
            self._neighbors[neighbor_id][layer] = self._select_neighbors(neighbor_vec, candidates, max_neighbors)

    def _dist_to(self, query: np.ndarray, node_id: str) -> float:
        return self._distance(query, self._vectors[node_id])

    def __len__(self) -> int:
        with self._lock:
            return len(self._vectors) - len(self._deleted)

    def stats(self) -> dict:
        with self._lock:
            return {
                "count": len(self._vectors) - len(self._deleted),
                "dim": self.dim,
                "metric": self.metric,
                "tombstoned": len(self._deleted),
                "entry_point": self._entry_point,
                "max_layer": self._entry_layer,
            }
