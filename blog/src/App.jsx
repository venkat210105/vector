import React, { useEffect, useMemo, useState } from "react";
import { marked } from "marked";
import Background from "./Background.jsx";
import { posts } from "./posts.js";

// Layout in the style of a docs/wiki site: page list on the left, the
// reading column in the middle, "On this page" outline on the right.

const slugify = (text) => text.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

// Give every heading an id so the outline can scroll to it.
marked.use({
  renderer: {
    heading(text, level) {
      return `<h${level} id="${slugify(text)}">${text}</h${level}>`;
    },
  },
});

// Tiny hash router: "#/" is the overview and "#/<slug>" is a post. Hash
// links need no server rules, so the site works on any static host.
function useSlug() {
  const read = () => window.location.hash.replace(/^#\/?/, "");
  const [slug, setSlug] = useState(read);
  useEffect(() => {
    const onChange = () => { setSlug(read()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return slug;
}

function Sidebar({ slug }) {
  return (
    <nav className="sidebar" aria-label="Pages">
      <a href="#/" className={`side-link ${slug === "" ? "active" : ""}`}>Overview</a>
      <div className="side-group">Notes</div>
      {posts.map((p, i) => (
        <a key={p.slug} href={`#/${p.slug}`} className={`side-link ${p.slug === slug ? "active" : ""}`}>
          <span className="side-num">{i + 1}</span>{p.title}
        </a>
      ))}
    </nav>
  );
}

// "On this page": the ## headings of the current post, with the one being read highlighted.
function Outline({ headings }) {
  const [active, setActive] = useState(headings[0]?.id);

  useEffect(() => {
    setActive(headings[0]?.id);
    const els = headings.map((h) => document.getElementById(h.id)).filter(Boolean);
    if (els.length === 0) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length) setActive(visible[0].target.id);
      },
      { rootMargin: "-80px 0px -70% 0px" }
    );
    els.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [headings]);

  if (headings.length === 0) return <aside className="outline" />;
  return (
    <aside className="outline" aria-label="On this page">
      <div className="outline-title">On this page</div>
      {headings.map((h) => (
        <button key={h.id} className={h.id === active ? "active" : ""} onClick={() => document.getElementById(h.id)?.scrollIntoView({ behavior: "smooth" })}>
          {h.text}
        </button>
      ))}
    </aside>
  );
}

function Overview() {
  return (
    <article>
      <h1>Inside a Vector Database</h1>
      <p className="lede">
        Plain-language notes on how a vector database works, written while building one from scratch in Python.
        Each page covers one idea and points at the real code behind it.
      </p>
      <ol className="post-list">
        {posts.map((p, i) => (
          <li key={p.slug}>
            <a href={`#/${p.slug}`}>
              <span className="num">{i + 1}</span>
              <span><b>{p.title}</b><br /><span className="muted">{p.summary}</span></span>
            </a>
          </li>
        ))}
      </ol>
    </article>
  );
}

function Post({ index, onHeadings }) {
  const post = posts[index];
  // The text is our own Markdown shipped with the site, so rendering it as HTML is safe.
  const html = useMemo(() => marked.parse(post.body), [post]);
  const headings = useMemo(
    () => marked.lexer(post.body).filter((t) => t.type === "heading" && t.depth === 2).map((t) => ({ text: t.text, id: slugify(t.text) })),
    [post]
  );
  useEffect(() => onHeadings(headings), [headings, onHeadings]);

  const prev = posts[index - 1];
  const next = posts[index + 1];
  return (
    <article>
      <p className="muted">Page {index + 1} of {posts.length}</p>
      <h1>{post.title}</h1>
      <div className="prose" dangerouslySetInnerHTML={{ __html: html }} />
      {post.Widget && <post.Widget />}
      {post.sources?.length > 0 && (
        <section className="sources">
          <h2>Sources</h2>
          <ul>
            {post.sources.map((s) => (
              <li key={s.path}><a href={s.url}>{s.path}</a></li>
            ))}
          </ul>
        </section>
      )}
      <nav className="pager">
        {prev ? <a href={`#/${prev.slug}`}><span className="muted">Previous</span><br />{prev.title}</a> : <span />}
        {next ? <a href={`#/${next.slug}`} className="next"><span className="muted">Next</span><br />{next.title}</a> : <span />}
      </nav>
    </article>
  );
}

export default function App() {
  const slug = useSlug();
  const index = posts.findIndex((p) => p.slug === slug);
  const [headings, setHeadings] = useState([]);

  useEffect(() => {
    document.title = index >= 0 ? `${posts[index].title} · Inside a Vector Database` : "Inside a Vector Database";
    if (index < 0) setHeadings([]);
  }, [index]);

  return (
    <div className="site">
      <Background />
      <header className="top">
        <a href="#/" className="brand">Inside a Vector Database</a>
        <a href="https://github.com/venkat210105/vector" className="repo-link">venkat210105/vector ↗</a>
      </header>
      <div className="layout">
        <Sidebar slug={index >= 0 ? posts[index].slug : ""} />
        <main>{index >= 0 ? <Post index={index} onHeadings={setHeadings} /> : <Overview />}</main>
        <Outline headings={headings} />
      </div>
    </div>
  );
}
