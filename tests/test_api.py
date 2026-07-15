import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VECTORDB_DATA_DIR", str(tmp_path))
    from vectordb.api import state

    state._registry = None
    from vectordb.api.main import app

    with TestClient(app) as test_client:
        yield test_client
    state._registry = None


class TestListCollections:
    def test_empty_registry_returns_empty_list(self, client: TestClient):
        response = client.get("/collections")
        assert response.status_code == 200
        assert response.json() == {"collections": []}

    def test_lists_created_collections_with_their_index_type(self, client: TestClient):
        client.post("/collections", json={"name": "flat_col", "dim": 4, "metric": "l2", "index_type": "flat"})
        client.post("/collections", json={"name": "hnsw_col", "dim": 4, "metric": "l2", "index_type": "hnsw"})

        response = client.get("/collections")
        assert response.status_code == 200
        collections = {c["name"]: c for c in response.json()["collections"]}

        assert collections["flat_col"]["index_type"] == "flat"
        assert collections["hnsw_col"]["index_type"] == "hnsw"
        assert collections["hnsw_col"]["dim"] == 4
        assert collections["hnsw_col"]["metric"] == "l2"
