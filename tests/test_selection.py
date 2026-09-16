import unittest

from flylab.selection import select_checkpoints


class SelectionTests(unittest.TestCase):
    def test_validation_only_and_prespecified_ties(self):
        candidates = [
            {'seed': 72, 'total_steps': 200, 'successes': 90, 'episodes': 100, 'test_successes': 99},
            {'seed': 71, 'total_steps': 100, 'successes': 90, 'episodes': 100, 'test_successes': 10},
            {'seed': 71, 'total_steps': 200, 'successes': 89, 'episodes': 100, 'test_successes': 100},
            {'seed': 72, 'total_steps': 100, 'successes': 90, 'episodes': 100, 'test_successes': 20},
        ]
        result = select_checkpoints(candidates)
        self.assertEqual(result['single_policy']['seed'], 71)
        self.assertEqual(result['single_policy']['total_steps'], 100)
        self.assertEqual(result['per_seed']['72']['total_steps'], 100)
        self.assertEqual(result, select_checkpoints(list(reversed(candidates))))

    def test_incomparable_or_duplicate_candidates_rejected(self):
        row = {'seed': 71, 'total_steps': 100, 'successes': 90, 'episodes': 100}
        for candidates in [[], [row, row], [row, {**row, 'seed': 72, 'episodes': 200}]]:
            with self.assertRaises(ValueError):
                select_checkpoints(candidates)
