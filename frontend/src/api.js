// Thin wrappers over the FastAPI backend. In dev, Vite proxies /api/* to it.
const BASE = "/api";

async function request(path, options = {}) {
  const response = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      // body wasn't JSON; keep statusText
    }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

const post = (path, body) => request(path, { method: "POST", body: JSON.stringify(body) });

export const listCollections = () => request("/collections").then((r) => r.collections);
export const createCollection = (body) => post("/collections", body);
export const upsertVector = (name, body) => post(`/collections/${name}/vectors`, body);
export const getStats = (name) => request(`/collections/${name}/stats`);
export const getGraph = (name, maxNodes = 500) => request(`/collections/${name}/graph?max_nodes=${maxNodes}`);
export const searchTrace = (name, body) => post(`/collections/${name}/search/trace`, body);
export const deleteVector = (name, id) => request(`/collections/${name}/vectors/${encodeURIComponent(id)}`, { method: "DELETE" });

// "Ask the project" talks to a separate small server (examples/rag/app.py),
// reached through the /rag proxy.
export async function askProject(question, k = 5) {
  const response = await fetch(`/rag/ask?q=${encodeURIComponent(question)}&k=${k}`);
  if (!response.ok) {
    let detail = null;
    try {
      detail = (await response.json()).detail;
    } catch {
      // no JSON body: the proxy could not reach the server at all
    }
    if (typeof detail === "string") throw new Error(detail);
    throw new Error("The ask server is not running. Start it with: python -m uvicorn examples.rag.app:app --port 8100");
  }
  return response.json();
}
