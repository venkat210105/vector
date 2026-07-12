import numpy as np
import pytest

from vectordb.core.flat_index import FlatIndex
from vectordb.core.hnsw.index import HNSWIndex
from vectordb.core.point import Point


def _build_indexes(n: int, dim: int, seed: int) -> tuple[HNSWIndex, FlatIndex]:
    """Builds an HNSWIndex and a FlatIndex over the same random vectors, so
    FlatIndex can serve as the ground-truth oracle HNSW's recall is checked
    against -- the same relationship the README describes between them."""
    rng = np.random.default_rng(seed)
    hnsw = HNSWIndex(dim=dim, seed=seed)
    flat = FlatIndex(dim=dim)
    for i in range(n):
        vector = rng.normal(size=dim).astype(np.float32)
        point = Point(id=f"p{i}", vector=vector)
        hnsw.insert(point)
        flat.upsert(point)
    return hnsw, flat


def _recall_at_k(hnsw: HNSWIndex, flat: FlatIndex, queries: list[np.ndarray], k: int, ef_search: int | None) -> float:
    hits = 0
    for query in queries:
        true_ids = {point_id for point_id, _ in flat.search(query, k)}
        found_ids = {point_id for point_id, _ in hnsw.search(query, k, ef_search=ef_search)}
        hits += len(true_ids & found_ids)
    return hits / (len(queries) * k)


class TestHNSWRecall:
    def test_recall_improves_as_ef_search_grows(self):
        hnsw, flat = _build_indexes(n=1000, dim=16, seed=7)
        rng = np.random.default_rng(99)
        queries = [rng.normal(size=16).astype(np.float32) for _ in range(20)]

        low_ef_recall = _recall_at_k(hnsw, flat, queries, k=10, ef_search=10)
        high_ef_recall = _recall_at_k(hnsw, flat, queries, k=10, ef_search=100)

        assert high_ef_recall >= low_ef_recall
        assert high_ef_recall >= 0.95

    def test_default_ef_search_gives_high_recall(self):
        hnsw, flat = _build_indexes(n=1000, dim=16, seed=7)
        rng = np.random.default_rng(123)
        queries = [rng.normal(size=16).astype(np.float32) for _ in range(20)]

        recall = _recall_at_k(hnsw, flat, queries, k=10, ef_search=None)
        assert recall >= 0.9

    def test_same_seed_gives_reproducible_graph(self):
        hnsw_a, _ = _build_indexes(n=200, dim=8, seed=42)
        hnsw_b, _ = _build_indexes(n=200, dim=8, seed=42)

        assert hnsw_a._entry_point == hnsw_b._entry_point
        assert hnsw_a._entry_layer == hnsw_b._entry_layer
        assert hnsw_a._neighbors == hnsw_b._neighbors


class TestHNSWSearchBasics:
    def test_empty_index_returns_no_results(self):
        hnsw = HNSWIndex(dim=4)
        assert hnsw.search(np.zeros(4, dtype=np.float32), k=5) == []

    def test_search_never_returns_more_than_k_results(self):
        hnsw, _ = _build_indexes(n=5, dim=4, seed=1)
        results = hnsw.search(np.zeros(4, dtype=np.float32), k=100)
        assert len(results) <= 5

    def test_search_results_are_sorted_nearest_first(self):
        hnsw, _ = _build_indexes(n=200, dim=8, seed=3)
        rng = np.random.default_rng(4)
        query = rng.normal(size=8).astype(np.float32)

        results = hnsw.search(query, k=10)
        distances = [dist for _, dist in results]
        assert distances == sorted(distances)


class TestHNSWInsert:
    def test_rejects_wrong_dimension_vector(self):
        hnsw = HNSWIndex(dim=4)
        with pytest.raises(ValueError):
            hnsw.insert(Point(id="p0", vector=np.zeros(3, dtype=np.float32)))

    def test_len_matches_number_of_inserted_points(self):
        hnsw, _ = _build_indexes(n=50, dim=4, seed=5)
        assert len(hnsw) == 50


class TestHNSWDelete:
    def test_delete_unknown_id_returns_false(self):
        hnsw = HNSWIndex(dim=4)
        assert hnsw.delete("nope") is False

    def test_delete_twice_returns_false_second_time(self):
        hnsw = HNSWIndex(dim=4, seed=1)
        hnsw.insert(Point(id="a", vector=np.zeros(4, dtype=np.float32)))
        assert hnsw.delete("a") is True
        assert hnsw.delete("a") is False

    def test_deleted_point_never_returned_by_search(self):
        hnsw, _ = _build_indexes(n=200, dim=8, seed=3)
        hnsw.delete("p0")

        rng = np.random.default_rng(4)
        for _ in range(10):
            query = rng.normal(size=8).astype(np.float32)
            found_ids = {pid for pid, _ in hnsw.search(query, k=50, ef_search=200)}
            assert "p0" not in found_ids

    def test_stats_reports_tombstoned_count(self):
        hnsw, _ = _build_indexes(n=10, dim=4, seed=2)
        hnsw.delete("p0")
        hnsw.delete("p1")

        stats = hnsw.stats()
        assert stats["tombstoned"] == 2
        assert stats["count"] == 8
        assert len(hnsw) == 8

    def test_reinsert_after_delete_makes_point_live_again(self):
        hnsw = HNSWIndex(dim=4, seed=1)
        hnsw.insert(Point(id="a", vector=np.zeros(4, dtype=np.float32)))
        hnsw.delete("a")

        hnsw.insert(Point(id="a", vector=np.ones(4, dtype=np.float32)))
        results = hnsw.search(np.ones(4, dtype=np.float32), k=1)
        assert results == [("a", 0.0)]

    def test_insert_works_after_deleting_the_only_point(self):
        """Regression test: deleting the sole point used to leave a dead
        entry_point with no live neighbors, crashing the next insert/search
        (IndexError from an empty _search_layer result)."""
        hnsw = HNSWIndex(dim=4, seed=1)
        hnsw.insert(Point(id="only", vector=np.zeros(4, dtype=np.float32)))
        hnsw.delete("only")
        assert len(hnsw) == 0

        hnsw.insert(Point(id="new", vector=np.ones(4, dtype=np.float32)))
        results = hnsw.search(np.ones(4, dtype=np.float32), k=1)
        assert results == [("new", 0.0)]

    def test_deleting_bridge_nodes_preserves_connectivity_and_recall(self):
        """The actual point of tombstoning over hard removal: deleting a
        chunk of the graph -- including nodes other nodes may route
        through -- must not disconnect the survivors from being found."""
        hnsw, flat = _build_indexes(n=500, dim=16, seed=11)
        rng = np.random.default_rng(12)

        to_delete = [f"p{i}" for i in range(0, 500, 10)]  # 50 of 500
        for point_id in to_delete:
            assert hnsw.delete(point_id) is True
            flat.delete(point_id)

        queries = [rng.normal(size=16).astype(np.float32) for _ in range(30)]
        hits = 0
        for query in queries:
            true_ids = {pid for pid, _ in flat.search(query, 10)}
            found = hnsw.search(query, k=10, ef_search=100)
            found_ids = {pid for pid, _ in found}
            assert found_ids.isdisjoint(to_delete)
            hits += len(true_ids & found_ids)

        assert hits / (len(queries) * 10) >= 0.9
