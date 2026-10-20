# Load test

Simulates many concurrent clients against the REST API with
[Locust](https://locust.io), and checks that concurrent writes survive a crash.

```powershell
pip install locust
python -m benchmarks.load_test.run_load_test
python -m benchmarks.load_test.run_load_test --users 10 50 --duration 10 --seed-size 500 --writes 500
```

| Option | Default | Meaning |
|---|---|---|
| `--users` | `10 50 100 200` | Simulated user counts, one stage each |
| `--duration` | `20` | Seconds per stage |
| `--seed-size` | `2000` | Vectors loaded before the test |
| `--writes` | `1000` | Concurrent writes in the correctness check |
| `--port` | `8001` | Port for the throwaway server |
| `--skip-correctness` | off | Only run the throughput sweep |

The server uses a temporary data directory that is deleted afterwards, so your
real `./data` folder is never touched.

## What it measures

1. **Throughput sweep:** requests per second, failure rate and p50/p95/p99
   latency per user count. `locustfile.py` defines the users: 4 searchers for
   every 1 writer.
2. **Correctness under concurrent writes:** sends exactly `--writes` unique
   upserts from 64 threads, checks the in-memory count, hard-kills the server
   (no clean shutdown), restarts it, and checks that WAL replay rebuilds the
   same count. A `MISMATCH` here means an acknowledged write was lost.

## Reading the results

- Locust and the server share this machine's CPU, so absolute numbers are
  lower than a dedicated load generator would show, and they vary from run to
  run. Compare stages within one run, and repeat before quoting a figure.
- Results and the open lost-write issue are written up in
  [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#load-test-simulated-concurrent-clients).
