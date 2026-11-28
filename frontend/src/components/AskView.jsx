import React, { useState } from "react";
import { askProject } from "../api.js";

const EXAMPLES = [
  "how does crash recovery work",
  "why does delete use tombstones",
  "what is the flush race",
  "how does HNSW search work",
];

// Semantic search over the project's own docs and code. The question is
// embedded, our HNSW index returns the closest chunks, and each is shown with
// a link to its source. Retrieval only: it shows the evidence, it does not
// write an answer.
export default function AskView() {
  const [question, setQuestion] = useState("");
  const [results, setResults] = useState(null);
  const [status, setStatus] = useState(null); // { kind: "busy" | "done" | "error", text }
  const [expanded, setExpanded] = useState(() => new Set()); // result indexes shown in full

  const run = async (text) => {
    const q = text.trim();
    if (!q) return;
    setStatus({ kind: "busy", text: "Searching…" });
    setResults(null);
    setExpanded(new Set());
    try {
      const data = await askProject(q);
      setResults(data.results);
      setStatus({ kind: "done", text: `${data.results.length} closest chunks` });
    } catch (e) {
      setStatus({ kind: "error", text: e.message });
    }
  };

  const toggle = (i) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });

  return (
    <div className="ask">
      <h1>Ask the project</h1>
      <p className="muted">Semantic search over this repo's docs and code, served by the vector database it describes.</p>

      <form
        className="ask-form"
        onSubmit={(e) => {
          e.preventDefault();
          run(question);
        }}
      >
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          aria-label="Your question about the project"
          placeholder="e.g. how does crash recovery work?"
          autoFocus
        />
        <button type="submit" className="primary" disabled={status?.kind === "busy"}>
          Search
        </button>
      </form>

      <div className="ask-chips">
        {EXAMPLES.map((text) => (
          <button
            key={text}
            type="button"
            className="chip"
            onClick={() => {
              setQuestion(text);
              run(text);
            }}
          >
            {text}
          </button>
        ))}
      </div>

      {status && <p className={`ask-status ${status.kind === "error" ? "error" : "muted"}`}>{status.text}</p>}

      {results?.map((r, i) => {
        const percent = Math.max(0, Math.min(100, Math.round(r.similarity * 100)));
        const long = r.text.split("\n").length > 10 || r.text.length > 700;
        return (
          <div className="ask-card" key={`${r.path}:${r.start_line}`}>
            <div className="ask-head">
              <a href={r.url} target="_blank" rel="noopener noreferrer">
                {r.path}:{r.start_line}
              </a>
              <span className="muted">similarity {r.similarity.toFixed(2)}</span>
            </div>
            <div className="ask-bar">
              <i style={{ width: `${percent}%` }} />
            </div>
            <pre className={expanded.has(i) ? "open" : ""}>{r.text}</pre>
            {long && (
              <button type="button" className="link" onClick={() => toggle(i)}>
                {expanded.has(i) ? "Show less" : "Show more"}
              </button>
            )}
          </div>
        );
      })}

      <p className="muted ask-foot">Retrieval only: these are the closest chunks, not a written answer.</p>
    </div>
  );
}
