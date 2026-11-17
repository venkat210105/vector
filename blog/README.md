# Inside a Vector Database (notes site)

A small static site with plain-language notes on how the vector database in
this repo works, with two interactive demos. No backend: everything runs in
the browser, so it can be hosted for free.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # static site in dist/
```

## Add a post

1. Write `src/posts/NN-title.md`.
2. Import it in `src/posts.js` and add an entry (an optional `Widget` shows an interactive demo after the text).

## Deploy on Netlify

Connect the repo. `netlify.toml` already sets the base folder (`blog`), the build command and the publish folder (`dist`).
