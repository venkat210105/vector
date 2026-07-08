import threading

import numpy as np

from vectordb.core.distance import DISTANCE_BATCH_FNS
from vectordb.core.point import Point


class FlatIndex:
    """Brute-force exact kNN index. Serves as the ground-truth baseline
    that HNSW's recall is measured against, and as the always-working
    fallback while HNSW is under development."""

    def __init__(self, dim: int, metric: str = "l2") -> None:
        if metric not in DISTANCE_BATCH_FNS:
            raise ValueError(f"unknown metric: {metric}")
        self.dim = dim
        self.metric = metric
        self._distance_batch = DISTANCE_BATCH_FNS[metric]
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._metadata: dict[str, dict] = {}
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._id_to_row: dict[str, int] = {}
        self._deleted_rows: set[int] = set()

    def upsert(self, point: Point) -> None:
        if point.vector.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {point.vector.shape}")
        with self._lock:
            if point.id in self._id_to_row:
                row = self._id_to_row[point.id]
                self._vectors[row] = point.vector
                self._metadata[point.id] = point.metadata
                self._deleted_rows.discard(row)
                return
            row = len(self._ids)
            self._ids.append(point.id)
            self._metadata[point.id] = point.metadata
            self._id_to_row[point.id] = row
            self._vectors = np.vstack([self._vectors, point.vector.reshape(1, -1)])

    def delete(self, point_id: str) -> bool:
        with self._lock:
            row = self._id_to_row.get(point_id)
            if row is None:
                return False
            self._deleted_rows.add(row)
            del self._id_to_row[point_id]
            return True

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        query = np.asarray(query, dtype=np.float32)
        with self._lock:
            if len(self._ids) == 0:
                return []
            distances = self._distance_batch(query, self._vectors)
            if self._deleted_rows:
                distances = distances.copy()
                distances[list(self._deleted_rows)] = np.inf
            k = min(k, len(self._ids) - len(self._deleted_rows))
            if k <= 0:
                return []
            top_rows = np.argpartition(distances, k - 1)[:k]
            top_rows = top_rows[np.argsort(distances[top_rows])]
            return [(self._ids[row], float(distances[row])) for row in top_rows]

    def __len__(self) -> int:
        with self._lock:
            return len(self._id_to_row)

    def stats(self) -> dict:
        with self._lock:
            return {
                "count": len(self._id_to_row),
                "dim": self.dim,
                "metric": self.metric,
                "tombstoned": len(self._deleted_rows),
            }
