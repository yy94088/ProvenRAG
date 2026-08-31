from __future__ import annotations

import re
from collections.abc import Iterable


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.casefold()))


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate suitable for an offline prototype."""

    return max(1, len(tokens(text)))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def shingles(text: str, size: int = 3) -> frozenset[tuple[str, ...]]:
    words = tokens(text)
    if len(words) < size:
        return frozenset((word,) for word in words)
    return frozenset(tuple(words[index : index + size]) for index in range(len(words) - size + 1))


def lexical_similarity(left: str, right: str) -> float:
    """Conservative near-duplicate score using words and ordered shingles."""

    word_score = jaccard(tokens(left), tokens(right))
    shingle_score = jaccard(shingles(left), shingles(right))
    return 0.35 * word_score + 0.65 * shingle_score

