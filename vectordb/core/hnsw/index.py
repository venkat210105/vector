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
    a few hops before refining locally. Insert/search algorithms land in
    follow-up commits — this is the graph representation and layer
    assignment they'll operate on.
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
