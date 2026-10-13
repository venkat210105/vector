import { createCollection, upsertVector } from "./api.js";

// A small clustered dataset so the graph has visible structure. Cluster
// centers come from a fixed seed, so queries can be generated later that
// land near the same clusters even after a page reload.
export const DEMO = { name: "demo", dim: 16, clusters: 8, points: 400, M: 8, efConstruction: 100 };

function mulberry32(seed) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const centers = (() => {
  const rng = mulberry32(42);
  return Array.from({ length: DEMO.clusters }, () => Array.from({ length: DEMO.dim }, () => gaussian(rng) * 3));
})();

const around = (cluster, rng, spread) => centers[cluster].map((c) => c + gaussian(rng) * spread);

export async function seedDemo(onProgress) {
  try {
    await createCollection({
      name: DEMO.name,
      dim: DEMO.dim,
      metric: "l2",
      index_type: "hnsw",
      M: DEMO.M,
      ef_construction: DEMO.efConstruction,
    });
  } catch (error) {
    if (error.status !== 409) throw error; // already exists: reuse it
    return;
  }
  const rng = mulberry32(7);
  // Sequential on purpose: every upsert fsyncs the WAL, and the collection
  // isn't built for concurrent writers yet.
  for (let i = 0; i < DEMO.points; i++) {
    const cluster = i % DEMO.clusters;
    await upsertVector(DEMO.name, { id: `d${i}`, vector: around(cluster, rng, 0.8), metadata: { cluster } });
    if (i % 20 === 0) onProgress?.(i / DEMO.points);
  }
  onProgress?.(1);
}

export function randomQuery(collection) {
  if (collection.name === DEMO.name) {
    return around(Math.floor(Math.random() * DEMO.clusters), Math.random, 1.2);
  }
  return Array.from({ length: collection.dim }, () => gaussian(Math.random));
}
