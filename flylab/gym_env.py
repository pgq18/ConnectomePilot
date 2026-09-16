"""Gymnasium residual-control interface for future PPO experiments.

The policy returns a normalized residual, not the whole robot action. With
features='fly', the reservoir is frozen; PPO sees its two output traces plus
goal/robot state. No range-sensor bypass is present in that observation.
"""
from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .world import NavigationWorld, Action
from .controllers import base_action, compose


def policy_observation(raw, brain=None):
    """Shared training/playback encoder; advances the brain exactly once."""
    neural = {}
    if brain is not None:
        _, neural = brain.act(raw)
        sensor = [min(1, neural["left_hz"]/50), min(1, neural["right_hz"]/50)]
    else:
        sensor = [r/NavigationWorld.ray_range for r in raw["rays"]]
    return np.asarray(sensor + [math.sin(raw["goal_angle"]), math.cos(raw["goal_angle"]),
                               min(1, raw["goal_distance"]/15), raw["linear"]/1.1,
                               raw["angular"]/2.2], dtype=np.float32), neural


class ResidualNavigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features="rays", scenario="clutter", data=None, brain_backend="scipy"):
        super().__init__()
        if features not in ("rays", "fly"):
            raise ValueError("features must be rays or fly")
        self.features, self.scenario = features, scenario
        self.brain = None
        if features == "fly":
            from .brain import FlyBrain, DATA
            self.brain = FlyBrain(data or DATA, backend=brain_backend)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, (14 if features == "rays" else 7,), dtype=np.float32)
        self.world = NavigationWorld()
        self._terminated = True

    def _observation(self, raw):
        observation, self.neural = policy_observation(raw, self.brain)
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # A seeded Gym reset fixes the next generated episode seed, while normal
        # resets keep sampling training scenes rather than repeating one map.
        # Training draws only from [0, 1M). Explicit evaluation scenes use a
        # disjoint seed range, so test layouts never enter training by chance.
        episode_seed = int((options or {}).get("scene_seed", self.np_random.integers(0, 1000000)))
        raw = self.world.reset(episode_seed, self.scenario)
        if self.brain:
            self.brain.reset(episode_seed)
        self._terminated = False
        return self._observation(raw), {"scene_seed": episode_seed}

    def step(self, action):
        if self._terminated:
            raise RuntimeError("Reset is required before stepping this episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Residual action must contain two finite values")
        action = np.clip(action, -1, 1)
        base = base_action(self.world.observe())
        residual = Action(float(action[0])*0.75, float(action[1])*1.8)
        final, bounded = compose(base, residual)
        raw, reward, terminated, truncated, info = self.world.step(final)
        # Penalize unnecessary corrections, including speed changes.
        reward -= 0.005*float(np.dot(action, action))
        self._terminated = terminated or truncated
        info.update({"base_action": [base.linear, base.angular],
                     "residual_action": [bounded.linear, bounded.angular],
                     "final_action": [final.linear, final.angular],
                     "scene_seed": self.world.seed})
        return self._observation(raw), reward, terminated, truncated, info
