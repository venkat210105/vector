"""The small API behind the "Ask the project" tab of the React explorer.

Run from the repo root, after `python -m examples.rag.ingest`:
    python -m uvicorn examples.rag.app:app --port 8100
and start the frontend as usual (`npm run dev` in frontend/). Vite proxies
/rag/* to this server.

It is read-only: it embeds a question, searches our own collection and returns
the closest chunks with links to the source on GitHub.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from examples.rag.embedder import Embedder
from examples.rag.ingest import DEFAULT_DATA_DIR, open_collection

SOURCE_URL = "https://github.com/venkat210105/vector/blob/master"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and the collection once, not on every question.
    app.state.embedder = Embedder()
    app.state.collection = open_collection(DEFAULT_DATA_DIR)
    yield
    app.state.collection._wal.close()


app = FastAPI(title="ask-the-project", lifespan=lifespan)


@app.get("/api/info")
def info() -> dict:
    return app.state.collection.stats()


@app.get("/api/ask")
def ask(q: str = Query(min_length=1, max_length=500), k: int = Query(5, ge=1, le=20)) -> dict:
    collection = app.state.collection
    query = app.state.embedder.embed_query(q)
    hits = collection.search(query, k=k)
    if not hits:
        raise HTTPException(404, "Nothing is stored yet. Run: python -m examples.rag.ingest")
    results = []
    for hit in hits:
        meta = hit["metadata"]
        results.append(
            {
                "path": meta["path"],
                "start_line": meta["start_line"],
                "similarity": round(1 - hit["distance"], 3),  # cosine distance -> similarity
                "text": meta["text"],
                "url": f"{SOURCE_URL}/{meta['path']}#L{meta['start_line']}",
            }
        )
    return {"question": q, "results": results}
