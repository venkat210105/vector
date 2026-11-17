"Close" needs a definition. This project supports two, and both return a **distance**: a number where **smaller means closer** and **0 means identical**. Keeping that one rule means the search code can always pick the smallest numbers, whichever metric a collection uses.

## L2: the straight line

L2 (Euclidean) distance is the straight-line distance between two points, the Pythagorean theorem extended to many dimensions. Small means close.

## Cosine: the angle

Cosine compares only the **direction** of two vectors, meaning the angle between them. It ignores how long they are. Embeddings often use it, because a longer text can produce a longer vector without meaning anything different.

The code returns `1 - cos(angle)`:

| Angle | cos(angle) | distance = 1 − cos(angle) | Meaning |
|---|---|---|---|
| 0° | 1 | **0** | same direction, most alike |
| 90° | 0 | **1** | unrelated |
| 180° | −1 | **2** | exact opposites |

The `1 −` flips "bigger means more similar" into "smaller means closer", so cosine follows the same rule as L2.

## Same direction, different length

Take `[1, 0]` and `[2, 0]`:

- L2 distance is **1**, because the points are one step apart.
- Cosine distance is **0**, because both point the same way.

Which metric you choose depends on whether length should matter. Try it below.

## One at a time, or all at once

Each metric has two forms in the code. One measures a single pair of vectors. The other, called `..._distance_to_all`, measures one query against **every** stored vector in a single numpy call. That is much faster than looping in Python, and it is what search uses.
