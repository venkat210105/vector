import React, { useEffect, useMemo, useState } from "react";
import { marked } from "marked";
import { posts } from "./posts.js";

// Tiny hash router: "#/" is the home page and "#/<slug>" is a post. Hash
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

function Home() {
  return (
    <>
      <h1>Inside a Vector Database</h1>
      <p className="lede">
        Plain-language notes on how a vector database works, written while building one from scratch in Python.
        Each post covers one idea, with the real code behind it.
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
    </>
  );
}

function Post({ index }) {
  const post = posts[index];
  // The post text is our own Markdown shipped with the site, so rendering it as HTML is safe.
  const html = useMemo(() => marked.parse(post.body), [post]);
  const prev = posts[index - 1];
  const next = posts[index + 1];
  return (
    <article>
      <p className="muted">Post {index + 1} of {posts.length}</p>
      <h1>{post.title}</h1>
      <div className="prose" dangerouslySetInnerHTML={{ __html: html }} />
      {post.Widget && <post.Widget />}
      <nav className="pager">
        {prev ? <a href={`#/${prev.slug}`}>← {prev.title}</a> : <span />}
        {next ? <a href={`#/${next.slug}`}>{next.title} →</a> : <span />}
      </nav>
    </article>
  );
}

export default function App() {
  const slug = useSlug();
  const index = posts.findIndex((p) => p.slug === slug);

  useEffect(() => {
    document.title = index >= 0 ? `${posts[index].title} · Inside a Vector Database` : "Inside a Vector Database";
  }, [index]);

  return (
    <div className="site">
      <header className="top">
        <a href="#/" className="brand">Inside a Vector Database</a>
        <a href="https://github.com/venkat210105/vector" className="muted">Source code ↗</a>
      </header>
      <main>{index >= 0 ? <Post index={index} /> : <Home />}</main>
      <footer className="muted">Notes from building a vector database from scratch.</footer>
    </div>
  );
}
