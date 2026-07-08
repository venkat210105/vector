import threading
from pathlib import Path

import numpy as np

from vectordb.core.flat_index import FlatIndex
from vectordb.core.point import Point
from vectordb.storage import snapshot
from vectordb.storage.recovery import recover
from vectordb.storage.wal import WAL

SNAPSHOT_EVERY_N_OPS = 1000


class Collection:
    """Ties a FlatIndex to its WAL + snapshot for crash-recoverable persistence.
    One Collection per named vector collection (analogous to a Pinecone index
    or a Postgres table)."""

    def __init__(self, name: str, dim: int, metric: str, data_dir: str | Path) -> None:
        self.name = name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wal_path = self.data_dir / f"{name}.wal"
        self.snapshot_path = self.data_dir / f"{name}.snapshot"

        self._wal = WAL(self.wal_path)
        self.index: FlatIndex = recover(dim, metric, self.snapshot_path, self._wal)
        self._ops_since_snapshot = 0
        self._snapshot_lock = threading.Lock()

    def upsert(self, point_id: str, vector: list[float] | np.ndarray, metadata: dict | None = None) -> None:
        vector = np.asarray(vector, dtype=np.float32)
        self._wal.append({"op": "upsert", "id": point_id, "vector": vector.tolist(), "metadata": metadata or {}})
        self.index.upsert(Point(id=point_id, vector=vector, metadata=metadata or {}))
        self._maybe_snapshot()

    def delete(self, point_id: str) -> bool:
        self._wal.append({"op": "delete", "id": point_id, "vector": None, "metadata": None})
        deleted = self.index.delete(point_id)
        self._maybe_snapshot()
        return deleted

    def search(self, query: list[float] | np.ndarray, k: int) -> list[dict]:
        query = np.asarray(query, dtype=np.float32)
        results = self.index.search(query, k)
        return [{"id": pid, "distance": dist, "metadata": self.index._metadata.get(pid, {})} for pid, dist in results]

    def stats(self) -> dict:
        return self.index.stats()

    def _maybe_snapshot(self) -> None:
        self._ops_since_snapshot += 1
        if self._ops_since_snapshot >= SNAPSHOT_EVERY_N_OPS:
            self.flush()

    def flush(self) -> None:
        """Forces a snapshot + WAL truncation. Also call on clean shutdown.

        The WAL is truncated immediately after the snapshot captures all
        prior records, so the snapshot's wal_offset is always 0: on
        recovery, the (now-truncated) WAL only ever contains records written
        after this snapshot.
        """
        with self._snapshot_lock:
            snapshot.save(self.index, self.snapshot_path, wal_offset=0)
            self._wal.truncate()
            self._ops_since_snapshot = 0

    def close(self) -> None:
        self.flush()
        self._wal.close()
