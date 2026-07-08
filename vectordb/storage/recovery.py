from pathlib import Path

import numpy as np

from vectordb.core.flat_index import FlatIndex
from vectordb.core.point import Point
from vectordb.storage import snapshot
from vectordb.storage.wal import WAL


def recover(dim: str, metric: str, snapshot_path: str | Path, wal: WAL) -> FlatIndex:
    """Loads the latest snapshot (if any) then replays WAL records written
    after the snapshot's offset. Used on process startup."""
    snapshot_path = Path(snapshot_path)
    if snapshot_path.exists():
        index, wal_offset = snapshot.load(snapshot_path)
    else:
        index = FlatIndex(dim=dim, metric=metric)
        wal_offset = 0

    for record in wal.read_from(wal_offset):
        if record["op"] == "upsert":
            index.upsert(Point(id=record["id"], vector=np.array(record["vector"], dtype=np.float32), metadata=record["metadata"] or {}))
        elif record["op"] == "delete":
            index.delete(record["id"])
    return index
