from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

from .models import Evidence, EvidenceGroup
from .similarity import lexical_similarity, tokens


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


class ProvenanceContractor:
    """Infer dependency components and contract each into one super-node."""

    def __init__(self, duplicate_threshold: float = 0.92) -> None:
        if not 0.0 <= duplicate_threshold <= 1.0:
            raise ValueError("duplicate_threshold must be in [0, 1]")
        self.duplicate_threshold = duplicate_threshold

    def contract(self, evidence: Iterable[Evidence]) -> tuple[EvidenceGroup, ...]:
        items = tuple(evidence)
        union_find = _UnionFind(len(items))

        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                if self._dependent(items[left], items[right]):
                    union_find.union(left, right)

        components: dict[int, list[Evidence]] = defaultdict(list)
        for index, item in enumerate(items):
            components[union_find.find(index)].append(item)

        groups = [self._make_group(members) for members in components.values()]
        return tuple(sorted(groups, key=lambda group: group.id))

    def _dependent(self, left: Evidence, right: Evidence) -> bool:
        if (
            left.provenance_hint is not None
            and right.provenance_hint is not None
            and left.provenance_hint == right.provenance_hint
        ):
            return True
        return lexical_similarity(left.text, right.text) >= self.duplicate_threshold

    @staticmethod
    def _make_group(members: list[Evidence]) -> EvidenceGroup:
        ordered = tuple(sorted(members, key=lambda item: item.id))
        # Stable tie-breaking is part of duplication-invariance: adding an
        # equal-scoring copy must not silently replace the serialized member.
        representative = min(ordered, key=lambda item: (-item.relevance, item.cost, item.id))
        hints = sorted({item.provenance_hint for item in ordered if item.provenance_hint})
        if hints:
            group_id = f"prov:{hints[0]}"
        else:
            normalized = " ".join(tokens(ordered[0].text))
            digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
            group_id = f"text:{digest}"
        slots = frozenset(slot for item in ordered for slot in item.slots)
        return EvidenceGroup(
            id=group_id,
            members=ordered,
            representative=representative,
            relevance=max(item.relevance for item in ordered),
            slots=slots,
        )
