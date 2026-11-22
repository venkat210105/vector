"""Split the project's own files into chunks small enough to embed.

A chunk is the unit that gets stored as one Point: a few paragraphs of a
markdown file, or one top-level function/class of a Python file. Chunks keep
their file path and starting line so an answer can cite its source.
"""

import re
from pathlib import Path

MAX_CHARS = 1500  # a block longer than this is split on blank lines
MIN_CHARS = 300   # tiny neighbouring blocks are merged up to this size

_PY_BOUNDARY = re.compile(r"^(class |def |async def |@)")
_MD_BOUNDARY = re.compile(r"^#{1,4} ")


def _split_blocks(lines: list[str], is_boundary) -> list[tuple[int, str]]:
    """Cut the file into blocks, starting a new block at every boundary line.
    Returns (1-based start line, text) pairs."""
    blocks, start, current = [], 0, []
    for i, line in enumerate(lines):
        # A decorator line belongs to the def right below it, so do not cut
        # between consecutive "@..." lines and their def.
        starts_new = is_boundary(line) and not (current and current[-1].startswith("@"))
        if starts_new and current:
            blocks.append((start + 1, "\n".join(current)))
            start, current = i, []
        current.append(line)
    if current:
        blocks.append((start + 1, "\n".join(current)))
    return blocks


def _split_long(start: int, text: str) -> list[tuple[int, str]]:
    """Break an over-long block on blank lines so each piece fits MAX_CHARS."""
    if len(text) <= MAX_CHARS:
        return [(start, text)]
    pieces, line_no, current, current_start = [], start, "", start
    for para in text.split("\n\n"):
        if current and len(current) + len(para) > MAX_CHARS:
            pieces.append((current_start, current))
            current, current_start = "", line_no
        current = f"{current}\n\n{para}" if current else para
        line_no += para.count("\n") + 2
    if current:
        pieces.append((current_start, current))
    return pieces


def _merge_small(blocks: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Glue a too-small block onto the next one so chunks carry enough context."""
    merged: list[tuple[int, str]] = []
    for start, text in blocks:
        if merged and len(merged[-1][1]) < MIN_CHARS:
            prev_start, prev_text = merged.pop()
            merged.append((prev_start, f"{prev_text}\n\n{text}"))
        else:
            merged.append((start, text))
    return merged


def chunk_text(text: str, suffix: str) -> list[tuple[int, str]]:
    """Chunk one file's text. `suffix` is ".md" or ".py"."""
    lines = text.splitlines()
    boundary = _MD_BOUNDARY.match if suffix == ".md" else _PY_BOUNDARY.match
    blocks = [piece for start, block in _split_blocks(lines, boundary) for piece in _split_long(start, block)]
    return [(s, t.strip()) for s, t in _merge_small(blocks) if t.strip()]


def project_files(root: Path) -> list[Path]:
    """The files worth asking questions about: docs, blog posts and source."""
    patterns = ["README.md", "docs/**/*.md", "blog/src/posts/*.md", "vectordb/**/*.py"]
    files = {p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
    return sorted(files)
