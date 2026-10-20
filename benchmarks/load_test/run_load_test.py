"""Sweeps simulated user counts against a throwaway vector-db server and
prints requests/sec, failures and latency percentiles for each stage.

    python -m benchmarks.load_test.run_load_test
    python -m benchmarks.load_test.run_load_test --users 10 50 100 --duration 20 --seed-size 2000

Safety: the server runs on its own port (default 8001) with a temporary data
directory that is deleted afterwards, so your real ./data folder is never read
or written.

Caveat: Locust and the server share this machine's CPU, so absolute numbers
are lower than a dedicated load generator would show. Compare stages with
each other, not with other people's benchmarks.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from vectordb.collection import Collection

HERE = Path(__file__).parent
COLLECTION = "load"
DIM = 16


def seed(data_dir: Path, n: int) -> None:
    """Builds the collection offline (no HTTP) and snapshots it, so the server
    starts with `n` vectors already loaded. Each upsert still fsyncs the WAL,
    so this takes a while for large `n`."""
    rng = np.random.default_rng(0)
    collection = Collection(COLLECTION, dim=DIM, metric="l2", data_dir=data_dir, index_type="hnsw", M=8, ef_construction=100)
    for i in range(n):
        collection.upsert(f"s{i}", rng.normal(size=DIM).tolist())
    collection.close()


def wait_until_up(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("server did not start in time")


def run_stage(users: int, duration: int, port: int, out_dir: Path) -> dict[str, dict]:
    """Runs Locust headless for one stage and returns its per-endpoint stats."""
    prefix = out_dir / f"u{users}"
    subprocess.run(
        [
            sys.executable, "-m", "locust",
            "-f", str(HERE / "locustfile.py"),
            "--headless",
            "--host", f"http://127.0.0.1:{port}",
            "-u", str(users),
            "-r", str(users),  # spawn all users within about a second
            "-t", f"{duration}s",
            "--csv", str(prefix),
            "--only-summary",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "LOAD_COLLECTION": COLLECTION, "LOAD_DIM": str(DIM)},
    )
    with open(f"{prefix}_stats.csv", newline="") as f:
        return {row["Name"]: row for row in csv.DictReader(f)}


def print_table(results: list[tuple[int, dict[str, dict]]]) -> None:
    header = f"{'users':>6} {'endpoint':<10} {'req/s':>8} {'fail %':>7} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}"
    print(header)
    print("-" * len(header))
    for users, stats in results:
        for name in ("search", "upsert", "Aggregated"):
            row = stats.get(name)
            if row is None:
                continue
            total = int(row["Request Count"]) or 1
            fail_pct = 100 * int(row["Failure Count"]) / total
            print(
                f"{users:>6} {name:<10} {float(row['Requests/s']):>8.1f} {fail_pct:>7.1f} "
                f"{float(row['50%']):>8.0f} {float(row['95%']):>8.0f} {float(row['99%']):>8.0f}"
            )
        print()


def start_server(port: int, data_dir: Path) -> subprocess.Popen:
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "vectordb.api.main:app", "--port", str(port), "--log-level", "warning"],
        env={**os.environ, "VECTORDB_DATA_DIR": str(data_dir)},
    )
    wait_until_up(port)
    return server


def vector_count(port: int) -> int:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/collections/{COLLECTION}/stats") as response:
        return json.load(response)["count"]


def upsert_one(port: int, point_id: str) -> int:
    body = json.dumps({"id": point_id, "vector": [0.1] * DIM, "metadata": {}}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/collections/{COLLECTION}/vectors",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.status


def check_concurrent_writes(port: int, data_dir: Path, writes: int = 1000, workers: int = 64) -> subprocess.Popen:
    """Correctness under concurrency, with no ambiguity: send exactly `writes`
    unique upserts from `workers` threads and wait for all of them. Then
    (1) the in-memory count must grow by exactly `writes`, and (2) after a
    hard kill (no clean shutdown, so no snapshot) and a restart, WAL replay
    must rebuild that same count. Returns the restarted server.

    Why not reuse the Locust numbers: when a Locust stage ends it abandons
    requests still in flight, so the server can finish writes that Locust
    never counted.
    """
    before = vector_count(port)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        statuses = list(pool.map(lambda i: upsert_one(port, f"c{i}"), range(writes)))
    after = vector_count(port)
    print(f"concurrent writes: {sum(s == 204 for s in statuses)}/{writes} returned 204")
    print(f"  in memory: {before} -> {after} (expected +{writes}) -> {'OK' if after - before == writes else 'MISMATCH'}")
    return after


def check_recovery(server: subprocess.Popen, port: int, data_dir: Path, expected: int) -> subprocess.Popen:
    server.kill()  # hard kill: skips the clean-shutdown snapshot, like kill -9
    server.wait(timeout=15)
    restarted = start_server(port, data_dir)
    recovered = vector_count(port)
    print(f"  after hard kill + restart: {recovered} (expected {expected}) -> {'OK' if recovered == expected else 'MISMATCH'}")
    return restarted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--users", type=int, nargs="+", default=[10, 50, 100, 200])
    parser.add_argument("--duration", type=int, default=20, help="seconds per stage")
    parser.add_argument("--seed-size", type=int, default=2000, help="vectors loaded before the test")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--writes", type=int, default=1000, help="concurrent writes in the correctness check")
    parser.add_argument("--skip-correctness", action="store_true", help="only run the throughput sweep")
    args = parser.parse_args()

    work_dir = Path(tempfile.mkdtemp(prefix="vectordb_load_"))
    data_dir = work_dir / "data"
    server = None
    try:
        print(f"seeding {args.seed_size} vectors in {data_dir} ...")
        seed(data_dir, args.seed_size)

        print(f"starting server on port {args.port} ...")
        server = start_server(args.port, data_dir)

        results = []
        for users in args.users:
            print(f"stage: {users} users for {args.duration}s ...")
            results.append((users, run_stage(users, args.duration, args.port, work_dir)))

        print()
        print_table(results)
        if not args.skip_correctness:
            expected = check_concurrent_writes(args.port, data_dir, writes=args.writes)
            server = check_recovery(server, args.port, data_dir, expected)
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=15)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
