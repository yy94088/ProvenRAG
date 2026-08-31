import unittest

from provenrag.copyburst import make_copyburst_case, run_copyburst
from provenrag.selectors import IndependentEvidenceSelector


class CopyBurstTests(unittest.TestCase):
    def test_contracted_result_is_duplication_invariant(self) -> None:
        selector = IndependentEvidenceSelector(budget=30)
        once = selector.select(make_copyburst_case(1))
        twenty = selector.select(make_copyburst_case(20))
        self.assertEqual(once.group_ids, twenty.group_ids)
        self.assertEqual(once.item_ids, twenty.item_ids)

    def test_copyburst_exposes_naive_failure(self) -> None:
        results = run_copyburst()
        by_key = {(result.burst, result.method): result for result in results}
        self.assertTrue(by_key[(1, "naive-density")].proxy_correct)
        self.assertFalse(by_key[(20, "naive-density")].proxy_correct)
        self.assertEqual(0.0, by_key[(20, "naive-density")].supporting_recall)

        for burst in (1, 5, 10, 20):
            result = by_key[(burst, "independent-density")]
            self.assertTrue(result.proxy_correct)
            self.assertEqual(1.0, result.supporting_recall)
            self.assertEqual(3, result.unique_provenance_groups)


if __name__ == "__main__":
    unittest.main()

