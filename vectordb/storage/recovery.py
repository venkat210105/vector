from pathlib import Path

import numpy as np

from vectordb.core.flat_index import FlatIndex
from vectordb.core.hnsw.index import HNSWIndex
from vectordb.core.point import Point
from vectordb.storage import snapshot
from vectordb.storage.wal import WAL

_INDEX_TYPES = {"flat": FlatIndex, "hnsw": HNSWIndex}


def recover(
    dim: int,
    metric: str,
    index_type: str,
    snapshot_path: str | Path,
    wal: WAL,
    M: int | None = None,
    ef_construction: int | None = None,
) -> FlatIndex | HNSWIndex:
    """Loads the latest snapshot (if any) then replays WAL records written
    after the snapshot's offset. Used on process startup. `index_type`
    (and `M`/`ef_construction`, HNSW-only) are only used to build a
    brand-new index when no snapshot exists yet -- once one exists, all of
    this is recovered from its own header, same as dim/metric already are.
    """
    snapshot_path = Path(snapshot_path)
    if snapshot_path.exists():
        index, wal_offset = snapshot.load(snapshot_path)
    else:
        if index_type not in _INDEX_TYPES:
            raise ValueError(f"unknown index_type: {index_type}")
        hnsw_kwargs = {}
        if index_type == "hnsw":
            if M is not None:
                hnsw_kwargs["M"] = M
            if ef_construction is not None:
                hnsw_kwargs["ef_construction"] = ef_construction
        index = _INDEX_TYPES[index_type](dim=dim, metric=metric, **hnsw_kwargs)
        wal_offset = 0

    for record in wal.read_from(wal_offset):
        if record["op"] == "upsert":
            point = Point(id=record["id"], vector=np.array(record["vector"], dtype=np.float32), metadata=record["metadata"] or {})
            if isinstance(index, HNSWIndex):
                index.insert(point)
            else:
                index.upsert(point)
        elif record["op"] == "delete":
            index.delete(record["id"])
    return index
