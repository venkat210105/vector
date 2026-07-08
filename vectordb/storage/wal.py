import os
import struct
import threading
from pathlib import Path
from typing import Iterator

import msgpack

_LEN_STRUCT = struct.Struct(">I")


class WAL:
    """Append-only write-ahead log. Each record is length-prefixed msgpack:
    {"op": "upsert"|"delete", "id": str, "vector": list[float] | None, "metadata": dict | None}

    fsync happens on every write (durable-but-slow); batching fsync across N
    writes or a timer is the documented throughput/durability tradeoff for v2.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = open(self.path, "ab")

    def append(self, record: dict) -> int:
        """Writes a record, fsyncs, and returns the byte offset it was written at."""
        payload = msgpack.packb(record, use_bin_type=True)
        with self._lock:
            offset = self._file.tell()
            self._file.write(_LEN_STRUCT.pack(len(payload)))
            self._file.write(payload)
            self._file.flush()
            os.fsync(self._file.fileno())
            return offset

    def read_from(self, offset: int = 0) -> Iterator[dict]:
        """Replays records starting at the given byte offset."""
        with open(self.path, "rb") as f:
            f.seek(offset)
            while True:
                len_bytes = f.read(_LEN_STRUCT.size)
                if len(len_bytes) < _LEN_STRUCT.size:
                    return
                (length,) = _LEN_STRUCT.unpack(len_bytes)
                payload = f.read(length)
                if len(payload) < length:
                    return
                yield msgpack.unpackb(payload, raw=False)

    def truncate(self) -> None:
        with self._lock:
            self._file.close()
            self._file = open(self.path, "wb")
            self._file.close()
            self._file = open(self.path, "ab")

    def size(self) -> int:
        with self._lock:
            return self._file.tell()

    def close(self) -> None:
        with self._lock:
            self._file.close()
