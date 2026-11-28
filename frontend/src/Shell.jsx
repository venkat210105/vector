import React, { useState } from "react";
import App from "./App.jsx";
import AskView from "./components/AskView.jsx";

const TABS = [
  { id: "explorer", label: "Explorer" },
  { id: "ask", label: "Ask the project" },
];

// A thin tab bar above the two views. Both views stay mounted and the hidden
// one is just not displayed, so switching tabs does not lose the graph or the
// last search.
export default function Shell() {
  const [tab, setTab] = useState("explorer");

  return (
    <>
      <nav className="topbar" aria-label="Views">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={tab === t.id ? "active" : ""}
            aria-current={tab === t.id ? "page" : undefined}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <div hidden={tab !== "explorer"}>
        <App />
      </div>
      <div hidden={tab !== "ask"}>
        <AskView />
      </div>
    </>
  );
}
