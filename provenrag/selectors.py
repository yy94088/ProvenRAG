from __future__ import annotations

from collections.abc import Iterable

from .models import Evidence, EvidenceGroup, Selection
from .provenance import ProvenanceContractor
from .similarity import lexical_similarity


class NaiveDenseSelector:
    """Document-level density baseline that counts duplicate support repeatedly."""

    def __init__(self, budget: int, alpha: float = 0.8, beta: float = 0.15) -> None:
        self.budget = budget
        self.alpha = alpha
        self.beta = beta

    def select(self, evidence: Iterable[Evidence]) -> Selection:
        remaining = list(evidence)
        selected: list[Evidence] = []
        covered: set[str] = set()
        spent = 0
        total_utility = 0.0

        while remaining:
            feasible = [item for item in remaining if spent + item.cost <= self.budget]
            if not feasible:
                break

            def rank(item: Evidence) -> tuple[float, float, str]:
                support = sum(lexical_similarity(item.text, prior.text) for prior in selected)
                coverage = len(item.slots - covered)
                gain = item.relevance + self.alpha * support + self.beta * coverage
                return gain / item.cost, item.relevance, item.id

            choice = max(feasible, key=rank)
            support = sum(lexical_similarity(choice.text, prior.text) for prior in selected)
            gain = choice.relevance + self.alpha * support + self.beta * len(choice.slots - covered)
            selected.append(choice)
            remaining.remove(choice)
            covered.update(choice.slots)
            spent += choice.cost
            total_utility += gain

        hints = tuple(
            item.provenance_hint or f"unresolved:{item.id}"
            for item in selected
        )
        return Selection(
            item_ids=tuple(item.id for item in selected),
            group_ids=hints,
            represented_evidence_ids=frozenset(item.id for item in selected),
            token_cost=spent,
            objective=total_utility / max(1, spent),
            covered_slots=frozenset(covered),
        )


class IndependentEvidenceSelector:
    """Budgeted greedy selection over provenance-contracted super-nodes."""

    def __init__(
        self,
        budget: int,
        *,
        alpha: float = 0.35,
        beta: float = 0.15,
        redundancy_penalty: float = 0.2,
        contractor: ProvenanceContractor | None = None,
    ) -> None:
        self.budget = budget
        self.alpha = alpha
        self.beta = beta
        self.redundancy_penalty = redundancy_penalty
        self.contractor = contractor or ProvenanceContractor()

    def select(self, evidence: Iterable[Evidence]) -> Selection:
        remaining = list(self.contractor.contract(evidence))
        selected: list[EvidenceGroup] = []
        covered: set[str] = set()
        spent = 0
        total_utility = 0.0

        while remaining:
            feasible = [group for group in remaining if spent + group.cost <= self.budget]
            if not feasible:
                break

            def gain(group: EvidenceGroup) -> float:
                similarities = [lexical_similarity(group.text, prior.text) for prior in selected]
                cross_group_support = sum(score for score in similarities if score >= 0.25)
                redundancy = sum(score * score for score in similarities)
                return (
                    group.relevance
                    + self.alpha * cross_group_support
                    + self.beta * len(group.slots - covered)
                    - self.redundancy_penalty * redundancy
                )

            choice = max(
                feasible,
                key=lambda group: (gain(group) / group.cost, group.relevance, group.id),
            )
            marginal = gain(choice)
            if marginal <= 0:
                break
            selected.append(choice)
            remaining.remove(choice)
            covered.update(choice.slots)
            spent += choice.cost
            total_utility += marginal

        represented = frozenset(
            member.id for group in selected for member in group.members
        )
        return Selection(
            item_ids=tuple(group.representative.id for group in selected),
            group_ids=tuple(group.id for group in selected),
            represented_evidence_ids=represented,
            token_cost=spent,
            objective=total_utility / max(1, spent),
            covered_slots=frozenset(covered),
        )

