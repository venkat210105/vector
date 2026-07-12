"""Benchmark harness: recall@k, latency percentiles, memory footprint, and
parameter sweeps for HNSWIndex, measured against FlatIndex (ground truth)
and faiss-cpu's HNSW implementation for credibility.

Run with (from the repo root, so `vectordb` is importable):
    python -m benchmarks.run_benchmark

Takes roughly 60-90s -- most of that is building this project's pure-Python
HNSWIndex twice (once for the main run, once again in an isolated
subprocess for the memory measurement) plus a small M-parameter sweep.
"""
import functools
import gc
import multiprocessing as mp
import os
import time

import faiss
import numpy as np
import psutil

from vectordb.core.flat_index import FlatIndex
from vectordb.core.hnsw.index import HNSWIndex
from vectordb.core.point import Point

N = 3000
DIM = 64
K = 10
SEED = 42
N_QUERIES = 30


def generate_dataset(n: int, dim: int, seed: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    ids = [f"p{i}" for i in range(n)]
    return ids, vectors


def build_flat(ids: list[str], vectors: np.ndarray) -> FlatIndex:
    index = FlatIndex(dim=vectors.shape[1])
    for pid, vec in zip(ids, vectors):
        index.upsert(Point(id=pid, vector=vec))
    return index


def build_hnsw(ids: list[str], vectors: np.ndarray, M: int = 16, ef_construction: int = 200, seed: int = 0) -> HNSWIndex:
    index = HNSWIndex(dim=vectors.shape[1], M=M, ef_construction=ef_construction, seed=seed)
    for pid, vec in zip(ids, vectors):
        index.insert(Point(id=pid, vector=vec))
    return index


def build_faiss_hnsw(vectors: np.ndarray, M: int = 16, ef_construction: int = 200) -> faiss.IndexHNSWFlat:
    index = faiss.IndexHNSWFlat(vectors.shape[1], M)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def percentiles(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.array(latencies_ms)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99))}


def measure_latency_ours(index, queries: np.ndarray, k: int, **search_kwargs) -> dict[str, float]:
    latencies = []
    for q in queries:
        start = time.perf_counter()
        index.search(q, k, **search_kwargs)
        latencies.append((time.perf_counter() - start) * 1000)
    return percentiles(latencies)


def measure_latency_faiss(index: faiss.IndexHNSWFlat, queries: np.ndarray, k: int, ef_search: int) -> dict[str, float]:
    index.hnsw.efSearch = ef_search
    latencies = []
    for q in queries:
        start = time.perf_counter()
        index.search(q.reshape(1, -1), k)
        latencies.append((time.perf_counter() - start) * 1000)
    return percentiles(latencies)


def recall_at_k(found_ids_per_query: list[list[str]], true_ids_per_query: list[list[str]], k: int) -> float:
    hits = sum(len(set(found) & set(true)) for found, true in zip(found_ids_per_query, true_ids_per_query))
    return hits / (len(true_ids_per_query) * k)


def _memory_worker(build_fn, queue: mp.Queue) -> None:
    """Runs in an isolated subprocess so one index's allocations can't
    contaminate the next measurement -- reports the RSS delta build_fn
    caused, not raw RSS (which would include interpreter/import overhead
    common to all three)."""
    gc.collect()
    proc = psutil.Process(os.getpid())
    baseline_mb = proc.memory_info().rss / (1024 * 1024)
    build_fn()
    gc.collect()
    after_mb = proc.memory_info().rss / (1024 * 1024)
    queue.put(after_mb - baseline_mb)


def measure_memory_mb(build_fn) -> float:
    queue = mp.Queue()
    p = mp.Process(target=_memory_worker, args=(build_fn, queue))
    p.start()
    result = queue.get()
    p.join()
    return result


def main() -> None:
    print(f"Dataset: {N} vectors, dim={DIM}, k={K}, {N_QUERIES} queries\n")
    rng = np.random.default_rng(SEED + 1)
    ids, vectors = generate_dataset(N, DIM, SEED)
    queries = rng.normal(size=(N_QUERIES, DIM)).astype(np.float32)

    print("Building FlatIndex (ground truth)...")
    flat = build_flat(ids, vectors)
    true_ids_per_query = [[pid for pid, _ in flat.search(q, K)] for q in queries]

    print("Building HNSWIndex (ours, M=16, ef_construction=200)...")
    t0 = time.perf_counter()
    hnsw = build_hnsw(ids, vectors, seed=SEED)
    hnsw_build_s = time.perf_counter() - t0
    print(f"  built in {hnsw_build_s:.1f}s ({hnsw_build_s / N * 1000:.2f} ms/insert)\n")

    print("Building faiss IndexHNSWFlat (M=16, ef_construction=200)...")
    t0 = time.perf_counter()
    faiss_index = build_faiss_hnsw(vectors)
    faiss_build_s = time.perf_counter() - t0
    print(f"  built in {faiss_build_s:.2f}s ({faiss_build_s / N * 1000:.3f} ms/insert)\n")

    print("=== Recall@k vs ef_search (ours) ===")
    print(f"{'ef_search':>10} | {'recall@10':>10}")
    for ef in [10, 20, 50, 100, 200, 400]:
        found = [[pid for pid, _ in hnsw.search(q, K, ef_search=ef)] for q in queries]
        r = recall_at_k(found, true_ids_per_query, K)
        print(f"{ef:>10} | {r:>10.2f}")
    print()

    chosen_ef = 100
    found = [[pid for pid, _ in hnsw.search(q, K, ef_search=chosen_ef)] for q in queries]
    ours_recall = recall_at_k(found, true_ids_per_query, K)

    faiss_index.hnsw.efSearch = chosen_ef
    _, faiss_result_indices = faiss_index.search(queries, K)
    faiss_found_ids = [[ids[i] for i in row if i != -1] for row in faiss_result_indices]
    faiss_recall = recall_at_k(faiss_found_ids, true_ids_per_query, K)

    print(f"=== Latency percentiles (ms/query, k={K}, ef_search={chosen_ef}) ===")
    flat_lat = measure_latency_ours(flat, queries, K)
    hnsw_lat = measure_latency_ours(hnsw, queries, K, ef_search=chosen_ef)
    faiss_lat = measure_latency_faiss(faiss_index, queries, K, ef_search=chosen_ef)
    print(f"{'index':>14} | {'recall@10':>10} | {'p50':>8} | {'p95':>8} | {'p99':>8}")
    print(f"{'FlatIndex':>14} | {'1.00':>10} | {flat_lat['p50']:>8.3f} | {flat_lat['p95']:>8.3f} | {flat_lat['p99']:>8.3f}")
    print(f"{'HNSW (ours)':>14} | {ours_recall:>10.2f} | {hnsw_lat['p50']:>8.3f} | {hnsw_lat['p95']:>8.3f} | {hnsw_lat['p99']:>8.3f}")
    print(f"{'faiss HNSW':>14} | {faiss_recall:>10.2f} | {faiss_lat['p50']:>8.3f} | {faiss_lat['p95']:>8.3f} | {faiss_lat['p99']:>8.3f}")
    print()

    print("=== Memory footprint (RSS delta, MB) ===")
    flat_mem = measure_memory_mb(functools.partial(build_flat, ids, vectors))
    hnsw_mem = measure_memory_mb(functools.partial(build_hnsw, ids, vectors, seed=SEED))
    faiss_mem = measure_memory_mb(functools.partial(build_faiss_hnsw, vectors))
    raw_vectors_mb = vectors.nbytes / (1024 * 1024)
    print(f"raw vector data: {raw_vectors_mb:.1f} MB ({N} x {DIM} x float32)")
    print(f"{'FlatIndex':>14} | {flat_mem:>8.1f} MB")
    print(f"{'HNSW (ours)':>14} | {hnsw_mem:>8.1f} MB")
    print(f"{'faiss HNSW':>14} | {faiss_mem:>8.1f} MB")
    print()

    print("=== Build time ===")
    print(f"{'HNSW (ours)':>14} | {hnsw_build_s:>8.1f}s | {hnsw_build_s / N * 1000:>6.2f} ms/insert")
    print(f"{'faiss HNSW':>14} | {faiss_build_s:>8.2f}s | {faiss_build_s / N * 1000:>6.3f} ms/insert")

    print("\n=== M sweep (N=1000, ef_construction=200, ef_search=100) ===")
    sweep_ids, sweep_vectors = ids[:1000], vectors[:1000]
    sweep_flat = build_flat(sweep_ids, sweep_vectors)
    sweep_true = [[pid for pid, _ in sweep_flat.search(q, K)] for q in queries]
    print(f"{'M':>6} | {'recall@10':>10} | {'build_s':>8}")
    for m in [8, 16, 32]:
        t0 = time.perf_counter()
        sweep_hnsw = build_hnsw(sweep_ids, sweep_vectors, M=m, seed=SEED)
        build_s = time.perf_counter() - t0
        found = [[pid for pid, _ in sweep_hnsw.search(q, K, ef_search=100)] for q in queries]
        r = recall_at_k(found, sweep_true, K)
        print(f"{m:>6} | {r:>10.2f} | {build_s:>8.1f}")


if __name__ == "__main__":
    main()
