import DistanceDemo from "./widgets/DistanceDemo.jsx";
import SearchDemo from "./widgets/SearchDemo.jsx";
import one from "./posts/01-what-is-a-vector-database.md?raw";
import two from "./posts/02-measuring-closeness.md?raw";
import three from "./posts/03-storing-points.md?raw";
import four from "./posts/04-deleting-and-searching.md?raw";
import five from "./posts/05-whats-next.md?raw";

// To add a post: drop a .md file in ./posts, import it above, add an entry
// here. `Widget` (optional) is an interactive demo shown after the text.
export const posts = [
  { slug: "what-is-a-vector-database", title: "What is a vector database?", summary: "Vectors, embeddings and the Point: what gets stored and what gets searched.", body: one },
  { slug: "measuring-closeness", title: "Measuring closeness: L2 and cosine", summary: "How the database decides that two vectors are similar.", body: two, Widget: DistanceDemo },
  { slug: "storing-points", title: "Storing points in a flat index", summary: "Rows, ids and why the vector size is checked.", body: three },
  { slug: "deleting-and-searching", title: "Deleting and searching", summary: "Tombstones, infinity and picking the k closest, step by step.", body: four, Widget: SearchDemo },
  { slug: "whats-next", title: "What these notes cover, and what's next", summary: "Where the foundation ends and the approximate index begins.", body: five },
];
