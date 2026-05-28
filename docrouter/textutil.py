"""Text utilities: HTML/XML stripping, chunking, and tokenization (stdlib only)."""
from __future__ import annotations

import re
from html.parser import HTMLParser

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "i", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "will", "with",
    "do", "does", "can", "could", "should", "would", "my", "we", "you", "your",
    "if", "about", "into", "than", "then", "there", "their", "they", "but", "not",
}

_SKIP_TAGS = {"script", "style", "noscript", "head", "header", "footer", "nav"}


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text from HTML/XML, skipping boilerplate tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(self._chunks)


def html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return clean_text(parser.text())


def strip_wiki_math(text: str) -> str:
    """Remove Wikipedia's leaked LaTeX rendering, e.g. '{\\displaystyle \\hat\\sigma}'.

    These braces are usually balanced and shallow, so a stack scan is enough.
    """
    if "{\\displaystyle" not in text and "\\displaystyle" not in text:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        # Detect the start of a '{\displaystyle ...}' group and skip it whole.
        if text[i] == "{" and text[i + 1 :].startswith("\\displaystyle"):
            depth = 0
            while i < n:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def clean_text(text: str) -> str:
    text = strip_wiki_math(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, target_words: int = 180, overlap_words: int = 40) -> list[str]:
    """Greedy word-window chunking with overlap. Sentence-aware at the boundaries."""
    text = clean_text(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.;:])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        current.append(sentence)
        count += len(words)
        if count >= target_words:
            chunks.append(" ".join(current))
            # Start the next chunk with a tail overlap for context continuity.
            tail = " ".join(current).split()[-overlap_words:]
            current = [" ".join(tail)] if overlap_words else []
            count = len(tail) if overlap_words else 0
    if current and (" ".join(current)).strip():
        chunks.append(" ".join(current))
    return [clean_text(c) for c in chunks if len(c.split()) > 5]


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
