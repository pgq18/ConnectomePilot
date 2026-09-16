"""Interactive goals: geometry, episode transitions, and controller comparisons."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from flylab.controllers import Controller
from flylab.server import Lab
from flylab.world import Action, NavigationWorld


class TargetTests(unittest.TestCase):
    def test_invalid_goals_preserve_the_entire_world(self):
        world = NavigationWorld()
        obstacle = world.obstacles[0]
        before = deepcopy(vars(world))
        candidates = [(float('nan'), 4), (2, float('inf')), (True, 4), ('2', 4),
                      (None, 4), ([], 4), (-1, 4), (12.1, 4), (2, .1),
                      (obstacle[0], obstacle[1]),
                      (obstacle[0] + obstacle[2] + world.radius / 2, obstacle[1])]
        for target in candidates:
            with self.subTest(target=target), self.assertRaises(ValueError):
                world.set_goal(*target)
            self.assertEqual(vars(world), before)

    def test_retarget_preserves_pose_and_map_but_starts_new_leg(self):
        world = NavigationWorld()
        for _ in range(5):
            world.step(Action(.4, .2))
        pose = (world.x, world.y, world.heading)
        obstacles = list(world.obstacles)
        observation = world.set_goal(2, 6)
        self.assertEqual((world.x, world.y, world.heading), pose)
        self.assertEqual(world.obstacles, obstacles)
        self.assertEqual(world.path, [pose[:2]])
        self.assertEqual((world.steps, world.path_length, world.reward), (0, 0, 0))
        self.assertEqual(world.last_action, Action(0, 0))
        self.assertAlmostEqual(observation['goal_distance'], math.hypot(2-world.x, 6-world.y))
        self.assertGreater(observation['goal_angle'], 0)

    def test_controller_reaches_an_arbitrary_goal(self):
        world = NavigationWorld()
        world.set_goal(1.7, 6)
        controller = Controller('base')
        while world.status == 'running':
            action, _ = controller.act(world.observe())
            world.step(action)
        self.assertEqual(world.status, 'success')
        self.assertLess(world.observe()['goal_distance'], .4)

    def test_terminal_transitions_and_goal_already_reached(self):
        for status in ['success', 'timeout']:
            world = NavigationWorld()
            world.status, world.steps = status, world.max_steps
            world.set_goal(2, 6)
            self.assertEqual((world.status, world.steps), ('running', 0))
            world.set_goal(world.x, world.y)
            self.assertEqual((world.status, world.steps), ('success', 0))
        world.status = 'collision'
        before = deepcopy(vars(world))
        with self.assertRaises(ValueError):
            world.set_goal(2, 6)
        self.assertEqual(vars(world), before)

    def test_benchmark_reset_still_uses_default_target(self):
        world = NavigationWorld(21, 'clutter')
        before = world.state()
        world.set_goal(1, 6)
        world.reset(21, 'clutter')
        self.assertEqual(world.state(), before)
        self.assertEqual(world.goal, (11.1, 4))


class LabTargetTests(unittest.TestCase):
    def test_goal_preserves_pause_state_and_clears_controller_telemetry(self):
        lab = Lab()
        for paused in [True, False]:
            lab.paused = paused
            lab.telemetry['final'] = [.8, .5]
            with patch.object(lab.controller, 'reset', wraps=lab.controller.reset) as reset:
                state = lab.command({'command': 'goal', 'x': 2, 'y': 6})
                reset.assert_called_once_with(lab.world.seed)
            self.assertEqual(state['paused'], paused)
            self.assertEqual(state['telemetry']['final'], [0, 0])
            self.assertEqual(state['world']['goal'], (2, 6))

    def test_same_map_resets_keep_target_and_new_maps_restore_default(self):
        lab = Lab()
        lab.command({'command': 'goal', 'x': 1.7, 'y': 6})
        lab.world.step(Action(.5, .2))
        state = lab.command({'command': 'reset', 'mode': 'base'})
        self.assertEqual(state['world']['goal'], (1.7, 6))
        self.assertEqual((state['world']['x'], state['world']['y']), (.8, 4))
        state = lab.command({'command': 'reset', 'seed': 2})
        self.assertEqual(state['world']['goal'], lab.world.default_goal)
        lab.command({'command': 'goal', 'x': 1.7, 'y': 6})
        state = lab.command({'command': 'reset', 'scenario': 'single'})
        self.assertEqual(state['world']['goal'], lab.world.default_goal)

    def test_rejected_goal_leaves_running_controller_untouched(self):
        lab = Lab()
        lab.command({'command': 'play'})
        before = lab.snapshot()
        with patch.object(lab.controller, 'reset') as reset:
            with self.assertRaises(ValueError):
                lab.command({'command': 'goal', 'x': -1, 'y': 4})
            reset.assert_not_called()
        self.assertEqual(lab.snapshot(), before)

    def test_retarget_after_arrival_and_clicking_current_pose(self):
        lab = Lab()
        lab.command({'command': 'play'})
        state = lab.command({'command': 'goal', 'x': lab.world.x, 'y': lab.world.y})
        self.assertEqual(state['world']['status'], 'success')
        self.assertTrue(state['paused'])
        state = lab.command({'command': 'goal', 'x': 1.7, 'y': 6})
        self.assertEqual(state['world']['status'], 'running')
        self.assertTrue(state['paused'])
        self.assertFalse(lab.command({'command': 'play'})['paused'])


if __name__ == '__main__':
    unittest.main()
