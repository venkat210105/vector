import os
import threading
from pathlib import Path

from fastapi import HTTPException

from vectordb.collection import Collection
from vectordb.storage import meta

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
        # A collection exists if it has a snapshot (older collections) or a
        # .meta file (written at creation, so WAL-only collections count too).
        names = {f.stem for pattern in ("*.snapshot", "*.meta") for f in self.data_dir.glob(pattern)}
        for name in sorted(names):
            self._reopen(name)

    def _reopen(self, name: str) -> Collection:
        # Prefer the creation parameters in <name>.meta. If it is missing, a
        # snapshot must exist (legacy collection) and carries them in its own
        # header, so the placeholders below are never used.
        meta_path = self.data_dir / f"{name}.meta"
        params = meta.load(meta_path) if meta_path.exists() else {"dim": 1, "metric": "l2", "index_type": "flat"}
        collection = Collection(
            name,
            dim=params["dim"],
            metric=params["metric"],
            data_dir=self.data_dir,
            index_type=params["index_type"],
            M=params.get("M"),
            ef_construction=params.get("ef_construction"),
        )
        self.collections[name] = collection
        return collection

    def create(
        self,
        name: str,
        dim: int,
        metric: str,
        index_type: str = "flat",
        M: int | None = None,
        ef_construction: int | None = None,
    ) -> Collection:
        with self._lock:
            collection = Collection(name, dim=dim, metric=metric, data_dir=self.data_dir, index_type=index_type, M=M, ef_construction=ef_construction)
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
