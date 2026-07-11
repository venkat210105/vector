from pathlib import Path

import numpy as np
import pytest

from vectordb.collection import Collection


def _make_hnsw_collection(tmp_path: Path, name: str = "col") -> Collection:
    return Collection(name, dim=4, metric="l2", data_dir=tmp_path, index_type="hnsw")


class TestCollectionHNSW:
    def test_upsert_and_search_round_trip(self, tmp_path: Path):
        collection = _make_hnsw_collection(tmp_path)
        collection.upsert("a", [1.0, 0.0, 0.0, 0.0], {"tag": "a"})
        collection.upsert("b", [0.0, 1.0, 0.0, 0.0], {"tag": "b"})
        collection.upsert("c", [0.9, 0.1, 0.0, 0.0], {"tag": "c"})

        results = collection.search([1.0, 0.0, 0.0, 0.0], k=2)

        assert [r["id"] for r in results] == ["a", "c"]
        assert results[0]["metadata"] == {"tag": "a"}

    def test_delete_is_rejected_with_clear_error(self, tmp_path: Path):
        collection = _make_hnsw_collection(tmp_path)
        collection.upsert("a", [1.0, 0.0, 0.0, 0.0])

        with pytest.raises(NotImplementedError):
            collection.delete("a")

    def test_survives_restart(self, tmp_path: Path):
        """The actual point of wiring HNSW into Collection: writes must
        survive a process restart, same guarantee FlatIndex already has."""
        collection = _make_hnsw_collection(tmp_path)
        for i in range(20):
            vec = np.zeros(4, dtype=np.float32)
            vec[i % 4] = float(i)
            collection.upsert(f"p{i}", vec.tolist())
        collection.close()

        reopened = _make_hnsw_collection(tmp_path)
        assert reopened.stats()["count"] == 20

        results = reopened.search([3.0, 0.0, 0.0, 0.0], k=1)
        assert len(results) == 1

    def test_survives_restart_via_snapshot(self, tmp_path: Path):
        """Same as above, but forces a snapshot (not just WAL replay) by
        crossing SNAPSHOT_EVERY_N_OPS, so the HNSW snapshot serialization
        path specifically gets exercised, not just WAL replay."""
        import vectordb.collection as collection_module

        original_threshold = collection_module.SNAPSHOT_EVERY_N_OPS
        collection_module.SNAPSHOT_EVERY_N_OPS = 5
        try:
            collection = _make_hnsw_collection(tmp_path, name="snap_col")
            for i in range(10):
                vec = np.zeros(4, dtype=np.float32)
                vec[i % 4] = float(i)
                collection.upsert(f"p{i}", vec.tolist())
            collection.close()

            reopened = _make_hnsw_collection(tmp_path, name="snap_col")
            assert reopened.stats()["count"] == 10
            assert reopened.index.__class__.__name__ == "HNSWIndex"
        finally:
            collection_module.SNAPSHOT_EVERY_N_OPS = original_threshold
