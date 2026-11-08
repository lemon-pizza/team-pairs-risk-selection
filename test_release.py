"""Prevent incomplete or differently dated replays from passing validation."""
import json
import unittest

import pandas as pd

from run_experiment import CASES, HERE, verify_results


class ReleaseValidationTests(unittest.TestCase):
    def setUp(self):
        self.results = pd.read_csv(HERE / 'reference_results.csv')
        self.reference = json.loads((HERE / 'reference.json').read_text())

    def test_complete_results(self):
        result = verify_results(self.results, self.reference, CASES)
        self.assertEqual(result['matched_cases'], 84)

    def test_missing_case_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing'):
            verify_results(self.results.iloc[:-1], self.reference, CASES)

    def test_wrong_calendar_rejected(self):
        self.results.loc[0, 'calendar_observations'] -= 1
        with self.assertRaisesRegex(ValueError, 'calendar_observations'):
            verify_results(self.results, self.reference, CASES)

    def test_partial_arm(self):
        base = self.results.loc[self.results.months.eq(6) & ~self.results.uncapped]
        result = verify_results(base, self.reference, [(6, False)])
        self.assertEqual(result['matched_cases'], 12)


if __name__ == '__main__':
    unittest.main()
