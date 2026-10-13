from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectordb.core.hnsw import viz
from vectordb.core.hnsw.index import HNSWIndex
from vectordb.core.point import Point


def _build(n: int = 200, dim: int = 8, seed: int = 3) -> HNSWIndex:
    rng = np.random.default_rng(seed)
    index = HNSWIndex(dim=dim, seed=seed)
    for i in range(n):
        index.insert(Point(id=f"p{i}", vector=rng.normal(size=dim).astype(np.float32)))
    return index


class TestSearchWithTrace:
    def test_results_match_plain_search(self):
        index = _build()
        query = np.random.default_rng(1).normal(size=8).astype(np.float32)
        plain = index.search(query, k=5, ef_search=40)
        traced, _ = index.search_with_trace(query, k=5, ef_search=40)
        assert traced == plain

    def test_trace_covers_every_layer_top_down_and_ends_at_layer_zero(self):
        index = _build()
        _, layers = index.search_with_trace(np.zeros(8, dtype=np.float32), k=5)
        assert [step["layer"] for step in layers] == list(range(index._entry_layer, -1, -1))
        assert layers[0]["entry"] == index._entry_point
        assert all(step["ef"] == 1 for step in layers[:-1])

    def test_each_expanded_node_is_new_and_discoveries_are_never_repeated(self):
        index = _build()
        _, layers = index.search_with_trace(np.zeros(8, dtype=np.float32), k=5, ef_search=30)
        for layer in layers:
            expanded = [s["expanded"] for s in layer["steps"]]
            assert len(expanded) == len(set(expanded))
            discovered = [d for s in layer["steps"] for d in s["discovered"]]
            assert len(discovered) == len(set(discovered))

    def test_empty_index_returns_nothing(self):
        results, layers = HNSWIndex(dim=4).search_with_trace(np.zeros(4, dtype=np.float32), k=3)
        assert results == [] and layers == []


class TestExportGraph:
    def test_nodes_edges_and_entry_point(self):
        index = _build(n=100)
        graph = viz.export_graph(index, max_nodes=500)
        assert len(graph["nodes"]) == 100
        assert graph["entry_point"] == index._entry_point
        assert graph["max_layer"] == index._entry_layer
        assert not graph["truncated"]
        ids = {n["id"] for n in graph["nodes"]}
        assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])

    def test_truncation_keeps_highest_layer_nodes(self):
        index = _build(n=200)
        graph = viz.export_graph(index, max_nodes=10)
        assert len(graph["nodes"]) == 10
        assert graph["truncated"] and graph["total_nodes"] == 200
        assert index._entry_point in {n["id"] for n in graph["nodes"]}

    def test_edges_are_deduplicated_per_layer(self):
        graph = viz.export_graph(_build(n=100))
        keys = [(e["layer"], e["source"], e["target"]) for e in graph["edges"]]
        assert len(keys) == len(set(keys))

    def test_tombstones_are_flagged(self):
        index = _build(n=50)
        index.delete("p3")
        nodes = {n["id"]: n for n in viz.export_graph(index)["nodes"]}
        assert nodes["p3"]["deleted"] and not nodes["p4"]["deleted"]

    def test_query_projection_lands_in_same_space_as_nodes(self):
        index = _build(n=100)
        nodes = {n["id"]: n for n in viz.export_graph(index)["nodes"]}
        x, y = viz.project_query(index, index._vectors["p7"])
        assert x == pytest.approx(nodes["p7"]["x"]) and y == pytest.approx(nodes["p7"]["y"])


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VECTORDB_DATA_DIR", str(tmp_path))
    from vectordb.api import state

    state._registry = None
    from vectordb.api.main import app

    with TestClient(app) as test_client:
        yield test_client
    state._registry = None


class TestVizEndpoints:
    def _make_hnsw(self, client: TestClient) -> None:
        client.post("/collections", json={"name": "h", "dim": 3, "metric": "l2", "index_type": "hnsw"})
        for i in range(20):
            client.post("/collections/h/vectors", json={"id": f"v{i}", "vector": [i, i % 3, 1.0]})

    def test_graph_and_trace_for_hnsw(self, client: TestClient):
        self._make_hnsw(client)
        graph = client.get("/collections/h/graph").json()
        assert len(graph["nodes"]) == 20

        response = client.post("/collections/h/search/trace", json={"vector": [3, 0, 1], "k": 3})
        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 3
        assert body["layers"][-1]["layer"] == 0
        assert len(body["query_xy"]) == 2

    def test_flat_collection_is_rejected_with_400(self, client: TestClient):
        client.post("/collections", json={"name": "f", "dim": 3, "metric": "l2", "index_type": "flat"})
        assert client.get("/collections/f/graph").status_code == 400
        assert client.post("/collections/f/search/trace", json={"vector": [0, 0, 0]}).status_code == 400

    def test_unknown_collection_is_404(self, client: TestClient):
        assert client.get("/collections/nope/graph").status_code == 404
