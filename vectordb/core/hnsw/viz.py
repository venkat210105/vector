"""Read-only helpers that turn an HNSWIndex into something drawable: a
2D projection of its vectors and a JSON-friendly dump of the layered graph.
Nothing here mutates the index."""

import numpy as np

from vectordb.core.hnsw.index import HNSWIndex


def _pca_basis(index: HNSWIndex) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean, basis) where basis is (2, dim): the top-2 principal
    directions of all stored vectors. Computed from the dim x dim covariance
    matrix, so cost is O(n * dim^2) regardless of how many points exist."""
    ids = list(index._vectors)
    mean = np.zeros(index.dim, dtype=np.float64)
    basis = np.zeros((2, index.dim), dtype=np.float64)
    if not ids:
        return mean, basis
    matrix = np.stack([index._vectors[i] for i in ids]).astype(np.float64)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    cov = centered.T @ centered / max(len(ids), 1)
    # eigh returns eigenvalues ascending; the last two columns are the top PCs.
    _, vectors = np.linalg.eigh(cov)
    top = vectors[:, ::-1][:, : min(2, index.dim)].T
    basis[: top.shape[0]] = top
    return mean, basis


def project_2d(index: HNSWIndex, vectors: np.ndarray, basis: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    mean, comps = basis if basis is not None else _pca_basis(index)
    return (np.atleast_2d(vectors).astype(np.float64) - mean) @ comps.T


def project_query(index: HNSWIndex, query: np.ndarray) -> list[float]:
    with index._lock:
        xy = project_2d(index, np.asarray(query, dtype=np.float32))[0]
    return [float(xy[0]), float(xy[1])]


def export_graph(index: HNSWIndex, max_nodes: int = 500) -> dict:
    """Dumps up to `max_nodes` nodes (highest layers first, so the sparse
    upper layers are always present) with 2D coordinates, plus every edge
    whose endpoints are both included. Edges are deduplicated per layer
    since neighbor lists are directed but visually undirected."""
    with index._lock:
        ids = sorted(index._vectors, key=lambda i: (-(len(index._neighbors[i]) - 1), i))[:max_nodes]
        included = set(ids)
        basis = _pca_basis(index)
        coords = project_2d(index, np.stack([index._vectors[i] for i in ids]), basis) if ids else np.zeros((0, 2))

        nodes = [
            {
                "id": nid,
                "level": len(index._neighbors[nid]) - 1,
                "deleted": nid in index._deleted,
                "x": float(coords[n, 0]),
                "y": float(coords[n, 1]),
                "metadata": index._metadata.get(nid, {}),
            }
            for n, nid in enumerate(ids)
        ]

        seen: set[tuple[int, str, str]] = set()
        edges: list[dict] = []
        for nid in ids:
            for layer, neighbors in enumerate(index._neighbors[nid]):
                for other in neighbors:
                    if other not in included:
                        continue
                    a, b = sorted((nid, other))
                    if (layer, a, b) in seen:
                        continue
                    seen.add((layer, a, b))
                    edges.append({"source": a, "target": b, "layer": layer})

        return {
            "nodes": nodes,
            "edges": edges,
            "entry_point": index._entry_point,
            "max_layer": index._entry_layer,
            "total_nodes": len(index._vectors),
            "truncated": len(index._vectors) > len(ids),
        }
