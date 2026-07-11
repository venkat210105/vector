from pathlib import Path

import msgpack
import numpy as np

from vectordb.core.flat_index import FlatIndex
from vectordb.core.hnsw.index import HNSWIndex


def save(index: FlatIndex | HNSWIndex, path: str | Path, wal_offset: int) -> None:
    """Serializes `index` to `path`. Both index types share the same
    on-disk envelope -- a length-prefixed msgpack header (which carries an
    `index_type` marker so `load` knows how to reconstruct it, same idea as
    how it already carries `dim`/`metric`) followed by a numpy vector blob.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(index, HNSWIndex):
        header, vectors = _hnsw_header(index, wal_offset)
    else:
        header, vectors = _flat_header(index, wal_offset)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        packed_header = msgpack.packb(header, use_bin_type=True)
        f.write(len(packed_header).to_bytes(8, "big"))
        f.write(packed_header)
        np.save(f, vectors, allow_pickle=False)
    tmp_path.replace(path)


def load(path: str | Path) -> tuple[FlatIndex | HNSWIndex, int]:
    path = Path(path)
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "big")
        header = msgpack.unpackb(f.read(header_len), raw=False)
        vectors = np.load(f, allow_pickle=False)

    if header.get("index_type", "flat") == "hnsw":
        return _load_hnsw(header, vectors)
    return _load_flat(header, vectors)


def _flat_header(index: FlatIndex, wal_offset: int) -> tuple[dict, np.ndarray]:
    with index._lock:
        header = {
            "index_type": "flat",
            "dim": index.dim,
            "metric": index.metric,
            "ids": index._ids,
            "metadata": index._metadata,
            "deleted_rows": list(index._deleted_rows),
            "wal_offset": wal_offset,
        }
        vectors = index._vectors.copy()
    return header, vectors


def _load_flat(header: dict, vectors: np.ndarray) -> tuple[FlatIndex, int]:
    index = FlatIndex(dim=header["dim"], metric=header["metric"])
    index._ids = header["ids"]
    index._metadata = header["metadata"]
    index._deleted_rows = set(header["deleted_rows"])
    index._vectors = vectors
    index._id_to_row = {pid: row for row, pid in enumerate(index._ids) if row not in index._deleted_rows}
    return index, header["wal_offset"]


def _hnsw_header(index: HNSWIndex, wal_offset: int) -> tuple[dict, np.ndarray]:
    with index._lock:
        ids = list(index._vectors.keys())
        header = {
            "index_type": "hnsw",
            "dim": index.dim,
            "metric": index.metric,
            "M": index.M,
            "ef_construction": index.ef_construction,
            "ids": ids,
            "metadata": index._metadata,
            "deleted": list(index._deleted),
            # positional, aligned with `ids` -- avoids duplicating id
            # strings as both list contents and dict keys on disk.
            "neighbors": [index._neighbors[pid] for pid in ids],
            "entry_point": index._entry_point,
            "entry_layer": index._entry_layer,
            "wal_offset": wal_offset,
        }
        vectors = np.stack([index._vectors[pid] for pid in ids]) if ids else np.empty((0, index.dim), dtype=np.float32)
    return header, vectors


def _load_hnsw(header: dict, vectors: np.ndarray) -> tuple[HNSWIndex, int]:
    index = HNSWIndex(dim=header["dim"], metric=header["metric"], M=header["M"], ef_construction=header["ef_construction"])
    ids = header["ids"]
    index._vectors = {pid: vectors[i] for i, pid in enumerate(ids)}
    index._metadata = header["metadata"]
    index._deleted = set(header["deleted"])
    index._neighbors = dict(zip(ids, header["neighbors"]))
    index._entry_point = header["entry_point"]
    index._entry_layer = header["entry_layer"]
    return index, header["wal_offset"]
