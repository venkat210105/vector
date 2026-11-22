"""Read the project's files, embed each chunk and store it in our own database.

Run from the repo root:
    python -m examples.rag.ingest

The collection lives in data/rag/ (git-ignored). It is rebuilt from scratch on
every run, so chunks from files that changed or were deleted never go stale.
"""

import argparse
from pathlib import Path

from examples.rag.chunking import chunk_text, project_files
from examples.rag.embedder import DIM, Embedder
from vectordb.collection import Collection

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "rag"
COLLECTION = "project"
BATCH = 32


def open_collection(data_dir: Path) -> Collection:
    return Collection(COLLECTION, dim=DIM, metric="cosine", data_dir=data_dir, index_type="hnsw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Start clean: the collection is derived data, so it is safe to rebuild.
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.data_dir.glob(f"{COLLECTION}.*"):
        stale.unlink()

    chunks = []  # (id, text, metadata)
    for path in project_files(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        for start_line, text in chunk_text(path.read_text(encoding="utf-8"), path.suffix):
            chunks.append((f"{rel}:{start_line}", text, {"path": rel, "start_line": start_line, "text": text}))
    print(f"{len(chunks)} chunks from {len({c[2]['path'] for c in chunks})} files")

    embedder = Embedder()
    collection = open_collection(args.data_dir)
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vectors = embedder.embed_passages([text for _, text, _ in batch])
        for (chunk_id, _, metadata), vector in zip(batch, vectors):
            collection.upsert(chunk_id, vector, metadata)
        print(f"  stored {min(i + BATCH, len(chunks))}/{len(chunks)}", end="\r")
    collection.close()
    print(f"\nDone. Stored in {args.data_dir}")


if __name__ == "__main__":
    main()
