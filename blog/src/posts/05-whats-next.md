These notes cover the foundation of the project: how items are represented, how closeness is measured, and how the exact (flat) index stores, deletes and searches.

A flat index checks every vector on every query, so it slows down as the data grows. The project also includes an approximate index (HNSW) that skips most of the data, and a write-ahead log so the database survives a crash. Those are the next topics, and they will be added to these notes as they are worked through.

## Look at the code

Every function mentioned here is short enough to read in one sitting:

- `vectordb/core/point.py`: the Point.
- `vectordb/core/distance.py`: L2 and cosine distance.
- `vectordb/core/flat_index.py`: storing, deleting and searching.

The repository also has a browser explorer that draws the approximate index as a graph and replays a search step by step.
