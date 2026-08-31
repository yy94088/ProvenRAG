import unittest

from provenrag.models import Evidence
from provenrag.provenance import ProvenanceContractor


class ProvenanceContractorTests(unittest.TestCase):
    def test_groups_near_duplicates_without_metadata(self) -> None:
        evidence = (
            Evidence("a", "Paris is the capital city of France.", 0.9),
            Evidence("b", "The capital city of France is Paris.", 0.9),
            Evidence("c", "Berlin is the capital city of Germany.", 0.9),
        )
        groups = ProvenanceContractor(duplicate_threshold=0.60).contract(evidence)
        member_sets = {frozenset(item.id for item in group.members) for group in groups}
        self.assertIn(frozenset({"a", "b"}), member_sets)
        self.assertIn(frozenset({"c"}), member_sets)

    def test_explicit_provenance_contracts_paraphrases(self) -> None:
        evidence = (
            Evidence("a", "One wording of the report.", 0.8, provenance_hint="origin"),
            Evidence("b", "A substantially rewritten report.", 0.9, provenance_hint="origin"),
        )
        groups = ProvenanceContractor().contract(evidence)
        self.assertEqual(1, len(groups))
        self.assertEqual("b", groups[0].representative.id)


if __name__ == "__main__":
    unittest.main()

