import os
import threading
from pathlib import Path

from fastapi import HTTPException

from vectordb.collection import Collection

DEFAULT_DATA_DIR = Path(os.environ.get("VECTORDB_DATA_DIR", "./data"))


class Registry:
    """In-process registry of open Collections. One process = one node,
    consistent with the v1 single-node scope."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.data_dir = data_dir
        self.collections: dict[str, Collection] = {}
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.data_dir.exists():
            return
        for snapshot_file in self.data_dir.glob("*.snapshot"):
            name = snapshot_file.stem
            self._reopen(name)

    def _reopen(self, name: str) -> Collection:
        # dim/metric/index_type are recovered from the snapshot header
        # itself; pass placeholders since recover() only uses them when no
        # snapshot exists.
        collection = Collection(name, dim=1, metric="l2", data_dir=self.data_dir, index_type="flat")
        self.collections[name] = collection
        return collection

    def create(self, name: str, dim: int, metric: str, index_type: str = "flat") -> Collection:
        with self._lock:
            collection = Collection(name, dim=dim, metric=metric, data_dir=self.data_dir, index_type=index_type)
            self.collections[name] = collection
            return collection

    def get_or_404(self, name: str) -> Collection:
        collection = self.collections.get(name)
        if collection is None:
            raise HTTPException(status_code=404, detail=f"collection '{name}' not found")
        return collection

    def close_all(self) -> None:
        for collection in self.collections.values():
            collection.close()


_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
