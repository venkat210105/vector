from pathlib import Path

import msgpack
import numpy as np

from vectordb.core.flat_index import FlatIndex


def save(index: FlatIndex, path: str | Path, wal_offset: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with index._lock:
        header = {
            "dim": index.dim,
            "metric": index.metric,
            "ids": index._ids,
            "metadata": index._metadata,
            "deleted_rows": list(index._deleted_rows),
            "wal_offset": wal_offset,
        }
        vectors = index._vectors.copy()

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        packed_header = msgpack.packb(header, use_bin_type=True)
        f.write(len(packed_header).to_bytes(8, "big"))
        f.write(packed_header)
        np.save(f, vectors, allow_pickle=False)
    tmp_path.replace(path)


def load(path: str | Path) -> tuple[FlatIndex, int]:
    path = Path(path)
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "big")
        header = msgpack.unpackb(f.read(header_len), raw=False)
        vectors = np.load(f, allow_pickle=False)

    index = FlatIndex(dim=header["dim"], metric=header["metric"])
    index._ids = header["ids"]
    index._metadata = header["metadata"]
    index._deleted_rows = set(header["deleted_rows"])
    index._vectors = vectors
    index._id_to_row = {pid: row for row, pid in enumerate(index._ids) if row not in index._deleted_rows}
    return index, header["wal_offset"]
