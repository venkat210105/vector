from fastapi import APIRouter, HTTPException

from vectordb.api.schemas import (
    CreateCollectionRequest,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    UpsertRequest,
)
from vectordb.api.state import get_registry

router = APIRouter()


@router.post("/collections", status_code=201)
def create_collection(req: CreateCollectionRequest):
    registry = get_registry()
    if req.name in registry.collections:
        raise HTTPException(status_code=409, detail=f"collection '{req.name}' already exists")
    registry.create(req.name, req.dim, req.metric, req.index_type)
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
    results = collection.search(req.vector, req.k)
    return {"results": results}


@router.get("/collections/{name}/stats", response_model=StatsResponse)
def stats(name: str):
    collection = get_registry().get_or_404(name)
    return collection.stats()


@router.get("/health")
def health():
    return {"status": "ok"}
