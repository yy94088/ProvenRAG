from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from .models import Evidence, Selection
from .selectors import IndependentEvidenceSelector, NaiveDenseSelector


_SLOTS = frozenset({"subject", "capital_relation", "answer"})
_GOLD_IDS = frozenset({"correct-a", "correct-b"})


@dataclass(frozen=True, slots=True)
class RunMetrics:
    burst: int
    method: str
    proxy_correct: bool
    supporting_recall: float
    selected_items: int
    unique_provenance_groups: int
    token_cost: int
    selected_ids: tuple[str, ...]


def make_copyburst_case(burst: int) -> tuple[Evidence, ...]:
    """Create one controlled correlated-evidence-amplification case.

    There are two independent correct sources and ``burst`` copies of one
    incorrect source. Labels exist only for evaluation.
    """

    if burst < 1:
        raise ValueError("burst must be at least 1")

    evidence = [
        Evidence(
            id="correct-a",
            text="Bavaria is a German state, and its capital is Munich.",
            relevance=0.94,
            provenance_hint="primary-a",
            slots=_SLOTS,
            label="support",
            token_cost=10,
        ),
        Evidence(
            id="correct-b",
            text="Munich is the capital and largest city of Bavaria.",
            relevance=0.92,
            provenance_hint="primary-b",
            slots=_SLOTS,
            label="support",
            token_cost=10,
        ),
    ]
    evidence.extend(
        Evidence(
            id=f"rumor-{index:02d}",
            text="Nuremberg is the capital of Bavaria.",
            relevance=0.97,
            provenance_hint="copied-rumor",
            slots=_SLOTS,
            label="distractor",
            token_cost=10,
        )
        for index in range(burst)
    )
    return tuple(evidence)


def _evaluate(method: str, burst: int, evidence: tuple[Evidence, ...], selection: Selection) -> RunMetrics:
    by_id = {item.id: item for item in evidence}
    selected = [by_id[item_id] for item_id in selection.item_ids]
    support_count = sum(item.label == "support" for item in selected)
    distractor_count = sum(item.label == "distractor" for item in selected)
    return RunMetrics(
        burst=burst,
        method=method,
        proxy_correct=support_count > distractor_count,
        supporting_recall=len(_GOLD_IDS & selection.represented_evidence_ids) / len(_GOLD_IDS),
        selected_items=len(selection.item_ids),
        unique_provenance_groups=len(set(selection.group_ids)),
        token_cost=selection.token_cost,
        selected_ids=selection.item_ids,
    )


def run_copyburst(bursts: tuple[int, ...] = (1, 5, 10, 20), budget: int = 30) -> tuple[RunMetrics, ...]:
    results: list[RunMetrics] = []
    for burst in bursts:
        evidence = make_copyburst_case(burst)
        naive = NaiveDenseSelector(budget=budget).select(evidence)
        independent = IndependentEvidenceSelector(budget=budget).select(evidence)
        results.append(_evaluate("naive-density", burst, evidence, naive))
        results.append(_evaluate("independent-density", burst, evidence, independent))
    return tuple(results)


def format_table(results: tuple[RunMetrics, ...]) -> str:
    lines = [
        "burst | method              | correct | support recall | groups | selected",
        "------|---------------------|---------|----------------|--------|------------------------------",
    ]
    for result in results:
        lines.append(
            f"{result.burst:>5} | {result.method:<19} | "
            f"{str(result.proxy_correct):<7} | {result.supporting_recall:>14.2f} | "
            f"{result.unique_provenance_groups:>6} | {','.join(result.selected_ids)}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic CopyBurst feasibility test")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    results = run_copyburst(budget=args.budget)
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))
    else:
        print(format_table(results))


if __name__ == "__main__":
    main()
