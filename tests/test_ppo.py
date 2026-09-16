"""Training interfaces and propagation checks; no biological claims from fixtures."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from flylab.gym_env import ResidualNavigationEnv


class TrainingSeedTests(unittest.TestCase):
    def test_explicit_evaluation_seed_and_training_range(self):
        env = ResidualNavigationEnv()
        a, _ = env.reset(seed=1, options={'scene_seed': 2000100})
        b, _ = env.reset(seed=55, options={'scene_seed': 2000100})
        self.assertEqual(env.world.seed, 2000100)
        np.testing.assert_array_equal(a, b)
        for _ in range(20):
            _, info = env.reset()
            self.assertTrue(0 <= info['scene_seed'] < 1000000)


class ActivePropagationTests(unittest.TestCase):
    def test_signed_sparse_matrix_matches_reference_exactly(self):
        from flylab.propagation import active_current
        rng = np.random.default_rng(234)
        matrix = sparse.random(257, 257, density=.07, random_state=rng, format='csr', dtype=np.float32)
        matrix.data = rng.uniform(-1, 1, matrix.nnz).astype(np.float32)
        columns = matrix.tocsc()
        for density in (0, .05, .5, 1):
            spikes = (rng.random(257) < density).astype(np.float32)
            actual = active_current(columns.indptr, columns.indices, columns.data, spikes)
            np.testing.assert_array_equal(actual, matrix @ spikes)

    def test_recurrent_state_and_actions_match_across_backends(self):
        from flylab.brain import FlyBrain
        from flylab.world import NavigationWorld
        # Synthetic six-cell fixture used only to test the simulator.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez(path/'brain.npz', cell_type=np.array(['LC4', 'LPLC2', 'DNp01']*2),
                     side=np.array(['L']*3+['R']*3))
            matrix = sparse.csr_matrix(([.5, .5, .5, .5], ([2, 2, 5, 5], [0, 1, 3, 4])), shape=(6, 6), dtype=np.float32)
            sparse.save_npz(path/'weights.npz', matrix)
            reference, active = FlyBrain(path), FlyBrain(path, backend='active')
            reference.reset(75); active.reset(75)
            raw = NavigationWorld().observe()
            for i in range(30):
                raw['rays'][2] = .3 if i % 2 else 3
                expected, _ = reference.act(raw)
                actual, _ = active.act(raw)
                self.assertEqual(expected, actual)
                for name in ('voltage', 'spikes', 'traces'):
                    np.testing.assert_array_equal(getattr(reference, name), getattr(active, name))


if __name__ == '__main__':
    unittest.main()
