The simplest index in the project is the **flat index**. It stores every Point and, to search, measures the query against all of them. It is slow on big data but always exact, so it is the baseline that faster, approximate indexes get compared against.

## What it holds

When a flat index is created, it keeps a few things in memory:

| Name | What it is |
|---|---|
| `dim` | How many numbers each vector has. Fixed for the whole index. |
| `metric` | Which distance to use, `"l2"` or `"cosine"`. |
| `_vectors` | A grid of numbers. **Each row is one stored vector.** |
| `_ids` | The id of each row, by position. |
| `_metadata` | The metadata, looked up by id. |
| `_id_to_row` | A lookup from an id to its row number. |

The same Point is spread across several of these. Its vector is in a row of `_vectors`, its name is in `_ids`, and its extras are in `_metadata`.

## Row, vector and id

The vector is the **data**, and the row is its **position** in the grid, like the seat number in a classroom. You ask for "row 7", not "the vector with these 384 numbers". Positions are cheap integers and are always unique, even if two Points happen to have equal vectors.

Ids are names, but the grid is addressed by position, so `_id_to_row` (for example `{"a": 0, "b": 1}`) lets the index jump from a name to its row.

## Upsert

`upsert` means *update if it exists, insert if it doesn't*:

1. **Check the size first.** If the vector's length is not `dim`, it raises an error and stores nothing, so a bad vector can never leave the index half-updated.
2. **If the id already exists,** overwrite that row's vector and replace its metadata.
3. **If it is new,** add the id, the metadata and the row-number link, and append the vector as a new row.

Nothing here measures any distance. `upsert` writes, and `search` reads.

## Why `dim` is checked

Distance is computed number by number, so two lists of different lengths cannot be compared. The grid also has a fixed number of columns. And a size mismatch is often a sign that two different embedding models are being mixed, which would make the results meaningless. Failing early turns that into a clear message.
