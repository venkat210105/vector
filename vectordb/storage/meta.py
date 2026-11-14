import json
import os
from pathlib import Path

# A collection's creation parameters. A snapshot carries these in its own
# header, but a collection that has not been snapshotted yet has only a WAL,
# and WAL records do not say what dim/metric/index type to rebuild. Writing
# this tiny file at creation time makes every collection recoverable from
# its very first write.


def save(path: str | Path, params: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(params, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)  # atomic: readers see the old file or the complete new one


def load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
