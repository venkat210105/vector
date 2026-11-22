"""Regression: a collection that was killed before its first snapshot
(only a .wal on disk) must come back after restart."""

from vectordb.api.state import Registry


def test_wal_only_flat_collection_survives_unclean_restart(tmp_path):
    registry = Registry(tmp_path)
    collection = registry.create("tiny", dim=3, metric="l2", index_type="flat")
    collection.upsert("a", [1.0, 0.0, 0.0], {"tag": "x"})
    # No close(): simulates kill -9 before any snapshot exists.

    reopened = Registry(tmp_path)

    assert list(reopened.collections) == ["tiny"]
    restored = reopened.collections["tiny"]
    assert restored.index.dim == 3
    assert restored.index.metric == "l2"
    results = restored.search([1.0, 0.0, 0.0], k=1)
    assert results[0]["id"] == "a"
    assert results[0]["metadata"] == {"tag": "x"}


def test_wal_only_hnsw_collection_keeps_its_parameters(tmp_path):
    registry = Registry(tmp_path)
    collection = registry.create("graph", dim=4, metric="cosine", index_type="hnsw", M=6, ef_construction=50)
    collection.upsert("a", [1.0, 0.0, 0.0, 0.0])
    collection.upsert("b", [0.0, 1.0, 0.0, 0.0])

    restored = Registry(tmp_path).collections["graph"]

    assert restored.index.M == 6
    assert restored.index.ef_construction == 50
    assert restored.index.metric == "cosine"
    assert restored.stats()["count"] == 2


def test_empty_collection_is_listed_after_restart(tmp_path):
    Registry(tmp_path).create("empty", dim=2, metric="l2")
    assert list(Registry(tmp_path).collections) == ["empty"]


def test_collection_with_snapshot_still_reopens(tmp_path):
    registry = Registry(tmp_path)
    collection = registry.create("snap", dim=2, metric="l2")
    collection.upsert("a", [1.0, 1.0])
    registry.close_all()  # clean shutdown writes a snapshot

    restored = Registry(tmp_path).collections["snap"]
    assert restored.stats()["count"] == 1
