## Deleting without moving anything

Deleting a Point does **not** remove its row from the grid. The index just notes that the row is dead. This is called a **tombstone**.

Why not remove the row?

1. **Cost.** Removing a row from a numpy grid copies the whole grid, because every later row shifts up.
2. **Correctness.** `_id_to_row` stores row numbers. If rows shifted, every id after the deleted one would point at the wrong vector. With a tombstone, no row ever moves, so no stored row number ever changes.

The price is that dead rows keep using memory, and search still measures distances to them. Cleaning them up for real is called **compaction**.

## Searching, step by step

Say we search with `k = 2`. We want the 2 nearest Points.

**1. Measure the query against every row.** One numpy call returns one distance per row, in row order.

```
row:       0     1     2     3     4
distance: 0.4   0.1   0.9   0.2   0.6
```

**2. Make deleted rows infinitely far.** If row 1 was deleted, its distance is set to `inf`, a value bigger than any real number. Search picks the *smallest* values, so a dead row can never win.

```
row:       0     1     2     3     4
distance: 0.4   inf   0.9   0.2   0.6
```

**3. Limit `k` to the live Points.** `k = min(k, number stored − number deleted)`. Without this, asking for more results than exist would fill the extra slots with deleted rows. If nothing is alive, search returns an empty list.

**4. Pick the k smallest.** Here that is rows 3 (0.2) and 0 (0.4). The code finds them with a fast partial step first and then sorts only those k, which is quicker than sorting every distance.

**5. Turn rows back into ids.** The winning row numbers are looked up in `_ids`, and each result is returned with its distance.

An empty index returns an empty list straight away, because there is nothing stored to compare against.

Try the five steps below. Delete rows, change `k`, and watch the result.
