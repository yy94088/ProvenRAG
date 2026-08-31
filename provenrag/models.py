from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Tuple

from .similarity import estimate_tokens


@dataclass(frozen=True, slots=True)
class Evidence:
    """One retrieved chunk or atomic claim.

    ``provenance_hint`` is optional metadata such as a canonical URL or known
    original article id. The MVP can also infer groups from near-duplicate text.
    ``label`` is used only by the synthetic evaluator, never by a selector.
    """

    id: str
    text: str
    relevance: float
    provenance_hint: str | None = None
    slots: FrozenSet[str] = field(default_factory=frozenset)
    label: str = "unknown"
    token_cost: int | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("evidence id must not be empty")
        if not self.text.strip():
            raise ValueError("evidence text must not be empty")
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError("relevance must be in [0, 1]")
        if self.token_cost is not None and self.token_cost <= 0:
            raise ValueError("token_cost must be positive")

    @property
    def cost(self) -> int:
        return self.token_cost or estimate_tokens(self.text)


@dataclass(frozen=True, slots=True)
class EvidenceGroup:
    """A super-node formed by contracting dependent evidence documents."""

    id: str
    members: Tuple[Evidence, ...]
    representative: Evidence
    relevance: float
    slots: FrozenSet[str]

    @property
    def cost(self) -> int:
        # Only one representative is serialized into the final context.
        return self.representative.cost

    @property
    def text(self) -> str:
        return self.representative.text


@dataclass(frozen=True, slots=True)
class Selection:
    """The selected context and enough metadata to audit it."""

    item_ids: Tuple[str, ...]
    group_ids: Tuple[str, ...]
    represented_evidence_ids: FrozenSet[str]
    token_cost: int
    objective: float
    covered_slots: FrozenSet[str]

