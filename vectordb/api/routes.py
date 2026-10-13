from fastapi import APIRouter, HTTPException, Query

from vectordb.api.schemas import (
    CreateCollectionRequest,
    ListCollectionsResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    UpsertRequest,
)
from vectordb.api.state import get_registry
from vectordb.core.hnsw.index import HNSWIndex

router = APIRouter()


@router.get("/collections", response_model=ListCollectionsResponse)
def list_collections():
    registry = get_registry()
    return {
        "collections": [
            {
                "name": name,
                "dim": collection.index.dim,
                "metric": collection.index.metric,
                "index_type": "hnsw" if isinstance(collection.index, HNSWIndex) else "flat",
            }
            for name, collection in registry.collections.items()
        ]
    }


@router.post("/collections", status_code=201)
def create_collection(req: CreateCollectionRequest):
    registry = get_registry()
    if req.name in registry.collections:
        raise HTTPException(status_code=409, detail=f"collection '{req.name}' already exists")
    registry.create(req.name, req.dim, req.metric, req.index_type, M=req.M, ef_construction=req.ef_construction)
    return {"name": req.name, "dim": req.dim, "metric": req.metric, "index_type": req.index_type}


@router.post("/collections/{name}/vectors", status_code=204)
def upsert_vector(name: str, req: UpsertRequest):
    collection = get_registry().get_or_404(name)
    if len(req.vector) != collection.index.dim:
        raise HTTPException(status_code=400, detail=f"expected vector of dim {collection.index.dim}, got {len(req.vector)}")
    collection.upsert(req.id, req.vector, req.metadata)


@router.delete("/collections/{name}/vectors/{point_id}")
def delete_vector(name: str, point_id: str):
    collection = get_registry().get_or_404(name)
    deleted = collection.delete(point_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"point '{point_id}' not found")
    return {"deleted": True}


@router.post("/collections/{name}/search", response_model=SearchResponse)
def search(name: str, req: SearchRequest):
    collection = get_registry().get_or_404(name)
    if len(req.vector) != collection.index.dim:
        raise HTTPException(status_code=400, detail=f"expected vector of dim {collection.index.dim}, got {len(req.vector)}")
    results = collection.search(req.vector, req.k, ef_search=req.ef_search)
    return {"results": results}


@router.post("/collections/{name}/search/trace")
def search_trace(name: str, req: SearchRequest):
    collection = get_registry().get_or_404(name)
    if len(req.vector) != collection.index.dim:
        raise HTTPException(status_code=400, detail=f"expected vector of dim {collection.index.dim}, got {len(req.vector)}")
    try:
        return collection.search_with_trace(req.vector, req.k, ef_search=req.ef_search)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/collections/{name}/graph")
def graph(name: str, max_nodes: int = Query(default=500, gt=0, le=5000)):
    collection = get_registry().get_or_404(name)
    try:
        return collection.graph(max_nodes=max_nodes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/collections/{name}/stats", response_model=StatsResponse)
def stats(name: str):
    collection = get_registry().get_or_404(name)
    return collection.stats()


@router.get("/health")
def health():
    return {"status": "ok"}
