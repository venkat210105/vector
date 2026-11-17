A normal database is good at exact questions: *give me the user with id 42*, or *every order over 100*. A **vector database** answers a different kind of question: *give me the things that are most **similar** to this one.*

## Vectors and embeddings

To compare similarity, everything is turned into a **vector**: a plain list of numbers, like `[0.12, -0.8, 0.33, ...]`. A model that has learned what text or pictures look like produces that list for any sentence or photo. The list is called an **embedding**.

The useful part: the model arranges things so that **similar meanings get close vectors**. Two sentences about the same topic produce lists that are near each other, and an unrelated sentence produces a list that is far away. After that, "find similar things" becomes "find the nearest vectors".

## What the database stores

Each stored item is a **Point** with three fields:

| Field | What it is | Example |
|---|---|---|
| `id` | A unique name, so you can find, update or delete the item later. | `"doc_42"` |
| `vector` | The embedding. **This is the only part search compares.** | `[0.12, -0.8, ...]` |
| `metadata` | Extra facts returned with a result. Search never looks at it. | `{"source": "faq.pdf"}` |

The photo or sentence itself is not inside the Point. The Point holds its vector and a name you can use to get back to the original.

## How a search works

1. You give the database one **query vector**, for example the embedding of a question someone just typed.
2. It measures how far that vector is from the stored ones.
3. It returns the `id` and `metadata` of the **k** closest.

This is the retrieval step in RAG. The database finds relevant passages, and a language model then answers using them. The database does not write the answer itself.

## How many numbers per vector?

The number of values in each vector is called `dim` (dimension). It is decided by the embedding model, not by the database. Many models use a few hundred or a thousand numbers. Every vector in one collection must have the same `dim`, because you cannot measure a distance between lists of different lengths.

The next post explains how "close" is measured.
