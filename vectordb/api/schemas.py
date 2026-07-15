from typing import Any

from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    name: str
    dim: int = Field(gt=0)
    metric: str = "l2"
    index_type: str = "flat"


class UpsertRequest(BaseModel):
    id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    vector: list[float]
    k: int = Field(default=10, gt=0)
    ef_search: int | None = Field(default=None, gt=0)


class SearchResult(BaseModel):
    id: str
    distance: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchResult]


class StatsResponse(BaseModel):
    count: int
    dim: int
    metric: str
    tombstoned: int
    entry_point: str | None = None
    max_layer: int | None = None


class CollectionSummary(BaseModel):
    name: str
    dim: int
    metric: str
    index_type: str


class ListCollectionsResponse(BaseModel):
    collections: list[CollectionSummary]
