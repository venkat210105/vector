"""Ask a question about this project and see the chunks the database finds.

Run from the repo root, after ingesting:
    python -m examples.rag.ask "why is the WAL written before the index?"

This is the retrieval half of RAG: the question is embedded, our own HNSW
index returns the k closest chunks, and each is printed with its file and line
so the answer can be checked against the source. The generation half (giving
these chunks to an LLM) is a separate step.
"""

import argparse
from pathlib import Path

from examples.rag.embedder import Embedder
from examples.rag.ingest import DEFAULT_DATA_DIR, open_collection

SNIPPET_CHARS = 400


def retrieve(question: str, k: int = 5, data_dir: Path = DEFAULT_DATA_DIR) -> list[dict]:
    """The k chunks closest to the question, nearest first."""
    collection = open_collection(data_dir)
    try:
        query = Embedder().embed_query(question)
        return collection.search(query, k=k)
    finally:
        collection._wal.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    for rank, hit in enumerate(retrieve(args.question, args.k, args.data_dir), start=1):
        meta = hit["metadata"]
        snippet = meta["text"][:SNIPPET_CHARS].replace("\n", "\n    ")
        print(f"\n[{rank}] {meta['path']}:{meta['start_line']}  (distance {hit['distance']:.3f})")
        print(f"    {snippet}")


if __name__ == "__main__":
    main()
