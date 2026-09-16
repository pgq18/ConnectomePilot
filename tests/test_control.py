import math
import tempfile
import unittest
from pathlib import Path

from flylab.world import NavigationWorld, Action
from flylab.controllers import Controller, base_action, compose


class WorldTests(unittest.TestCase):
    def test_seeded_episodes_match(self):
        a, b = NavigationWorld(234, "clutter"), NavigationWorld(234, "clutter")
        self.assertEqual(a.state(), b.state())
        for _ in range(20):
            action = Action(.4, .3)
            self.assertEqual(a.step(action), b.step(action))
        self.assertEqual(a.state(), b.state())

    def test_straight_drive_hits_obstacle(self):
        w = NavigationWorld(1, "single")
        while w.status == "running":
            w.step(Action(1.1, 0))
        self.assertEqual(w.status, "collision")
        self.assertLess(w.steps, 100)
        with self.assertRaises(RuntimeError):
            w.step(Action(0, 0))

    def test_goal_and_timeout_are_distinct(self):
        w = NavigationWorld()
        w.goal = (w.x+.1, w.y)
        _, _, terminated, truncated, _ = w.step(Action(0, 0))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        w.reset()
        w.steps = w.max_steps-1
        _, _, terminated, truncated, _ = w.step(Action(0, 0))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_residual_zero_gain_restores_base(self):
        base = base_action(NavigationWorld().observe())
        actual, residual = compose(base, Action(-100, 100), 0)
        self.assertEqual(actual, base.bounded())
        self.assertEqual(residual, Action(0, 0))

    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            NavigationWorld().step(Action(float("nan"), 0))

    def test_missing_brain_never_falls_back(self):
        with self.assertRaises(ValueError):
            Controller("fly")


class BrainTests(unittest.TestCase):
    def test_synthetic_fixture_validates_matrix_direction(self):
        # This SIX-node artificial fixture is solely an integration test.
        # It is never used as downloadable data or exposed as a fly simulation.
        import numpy as np
        from scipy import sparse
        from flylab.brain import FlyBrain
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            types = np.array(["LC4", "LPLC2", "DNp01", "LC4", "LPLC2", "DNp01"])
            np.savez(p/"brain.npz", cell_type=types, side=np.array(["L"]*3+["R"]*3))
            weights = sparse.csr_matrix(([.5,.5,.5,.5], ([2,2,5,5], [0,1,3,4])), shape=(6,6))
            sparse.save_npz(p/"weights.npz", weights)
            b = FlyBrain(p)
            obs = NavigationWorld().observe()
            obs["rays"] = [3]*9
            obs["rays"][7] = .2
            _, output = b.act(obs)
            self.assertGreater(output["left_hz"], output["right_hz"])
            self.assertEqual(output["neurons"], 6)
            b.reset(1)
            _, again = b.act(obs)
            self.assertEqual(output["left_hz"], again["left_hz"])


class GymTests(unittest.TestCase):
    def test_gym_contract(self):
        from gymnasium.utils.env_checker import check_env
        from flylab.gym_env import ResidualNavigationEnv
        env = ResidualNavigationEnv()
        check_env(env, skip_render_check=True)


class DataPreparationTests(unittest.TestCase):
    def test_official_schema_filters_non_neurons_and_preserves_direction(self):
        # Deliberately synthetic data exercising the official table schema.
        # This fixture stays in a temporary test directory.
        import numpy as np
        import pyarrow as pa
        import pyarrow.feather as feather
        from scipy import sparse
        from scripts.prepare_data import build_official, FILES
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            raw = p/"raw"
            raw.mkdir()
            feather.write_feather(pa.Table.from_pylist([
                {"bodyId": 10, "superclass": "visual_projection", "type": "LC4", "somaSide": "L", "instance": "LC4_L"},
                {"bodyId": 20, "superclass": "descending", "type": "DNp01", "somaSide": "L", "instance": "DNp01_L"},
                {"bodyId": 30, "superclass": None, "type": None, "somaSide": None, "instance": None}
            ]), raw/FILES["annotations"])
            feather.write_feather(pa.Table.from_pylist([
                {"body": 10, "consensus_nt": "acetylcholine"},
                {"body": 20, "consensus_nt": "GABA"}
            ]), raw/FILES["transmitters"])
            feather.write_feather(pa.Table.from_pylist([
                {"body_pre": 10, "body_post": 20, "weight": 8},
                {"body_pre": 20, "body_post": 10, "weight": 4},
                {"body_pre": 30, "body_post": 20, "weight": 100}
            ]), raw/FILES["connections"])
            build_official(p)
            w = sparse.load_npz(p/"weights.npz")
            self.assertEqual(w.shape, (2, 2))
            self.assertEqual(w.nnz, 2)
            self.assertEqual(float(w[1, 0]), 1.0)
            self.assertEqual(float(w[0, 1]), -1.0)
            meta = np.load(p/"brain.npz", allow_pickle=False)
            self.assertEqual(meta["ids"].tolist(), [10, 20])


class LabTests(unittest.TestCase):
    def test_commands_reset_controller_and_scene_together(self):
        from flylab.server import Lab
        lab = Lab()
        state = lab.command({"command": "reset", "mode": "base", "seed": 99,
                             "scenario": "single", "gain": 0})
        self.assertEqual(state["mode"], "base")
        self.assertEqual(state["world"]["seed"], 99)
        self.assertTrue(state["paused"])
        self.assertEqual(state["gain"], 0)
        self.assertFalse(lab.command({"command": "play"})["paused"])
        self.assertTrue(lab.command({"command": "pause"})["paused"])

    def test_rejected_reset_preserves_current_episode(self):
        from flylab.server import Lab
        lab = Lab()
        before = lab.snapshot()
        with self.assertRaises(ValueError):
            lab.command({"command": "reset", "mode": "not-a-mode", "seed": 2})
        self.assertEqual(lab.snapshot()["world"], before["world"])


if __name__ == "__main__":
    unittest.main()
