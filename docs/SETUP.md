# Setup & Running Locally

## First-time setup

The venv already exists at `.venv`, but the API dependencies
(`fastapi`/`pydantic`/`uvicorn`/`httpx`) may not be installed yet — they're
declared in `requirements.txt` but installing that file is a separate step.

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `Activate.ps1` is blocked by execution policy, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## Start the server

```powershell
uvicorn vectordb.api.main:app --reload --port 8000
```

`--reload` restarts the server automatically when you edit code. Data is
written under `./data` by default (override with the `VECTORDB_DATA_DIR`
env var).

## Try it — interactive docs (easiest)

Once the server's running, open **http://127.0.0.1:8000/docs** in a
browser. FastAPI auto-generates a Swagger UI page there — click any
endpoint, hit "Try it out," fill in the JSON body, and execute it directly
from the browser. No separate tool needed.

## Try it — from the command line

Run these in a **second** PowerShell window (the first is busy running the
server):

```powershell
# create a collection, HNSW-backed
curl.exe -X POST http://127.0.0.1:8000/collections `
  -H "Content-Type: application/json" `
  -d '{\"name\": \"demo\", \"dim\": 4, \"metric\": \"l2\", \"index_type\": \"hnsw\"}'

# upsert a couple vectors
curl.exe -X POST http://127.0.0.1:8000/collections/demo/vectors `
  -H "Content-Type: application/json" `
  -d '{\"id\": \"a\", \"vector\": [1,0,0,0], \"metadata\": {\"tag\": \"a\"}}'

curl.exe -X POST http://127.0.0.1:8000/collections/demo/vectors `
  -H "Content-Type: application/json" `
  -d '{\"id\": \"b\", \"vector\": [0,1,0,0]}'

# search
curl.exe -X POST http://127.0.0.1:8000/collections/demo/search `
  -H "Content-Type: application/json" `
  -d '{\"vector\": [1,0,0,0], \"k\": 2}'

# stats -- note entry_point/max_layer only populate for hnsw collections
curl.exe http://127.0.0.1:8000/collections/demo/stats

# delete -- expect 501 for hnsw (not yet supported), 200 for a flat collection
curl.exe -X DELETE http://127.0.0.1:8000/collections/demo/vectors/a
```

To try the same thing with the brute-force baseline instead, create a
collection with `"index_type": "flat"` (or omit it — that's the default)
and delete will actually succeed instead of returning `501`.

## Run the test suite

```powershell
pytest tests/ -v
```

## Restart-survival check (proves persistence actually works)

1. Start the server, create a collection, upsert some vectors.
2. Stop the server (Ctrl+C).
3. Start it again with the same command.
4. Search or fetch stats on the same collection — the data is still there,
   loaded from `./data/<name>.snapshot` + `./data/<name>.wal` on startup.
