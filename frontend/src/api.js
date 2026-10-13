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
