"""Deterministic differential-drive navigation. No brain dependency."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


def clip(value, low, high):
    return max(low, min(high, value))


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Action:
    linear: float
    angular: float

    def bounded(self):
        return Action(clip(self.linear, 0.0, 1.1), clip(self.angular, -2.2, 2.2))


class NavigationWorld:
    width, height, radius, dt, max_steps = 12.0, 8.0, 0.19, 0.1, 600
    ray_angles = tuple(math.radians(a) for a in (-90, -65, -40, -20, 0, 20, 40, 65, 90))
    ray_range = 3.0

    def __init__(self, seed=1, scenario="slalom"):
        self.reset(seed, scenario)

    def reset(self, seed=1, scenario="slalom"):
        if scenario not in ("slalom", "clutter", "single"):
            raise ValueError("Unknown scenario")
        self.seed, self.scenario = int(seed), scenario
        rng = random.Random(self.seed)
        self.x, self.y, self.heading = 0.8, 4.0, 0.0
        self.goal = (11.1, 4.0)
        if scenario == "single":
            self.obstacles = [(5.5, 4.0 + rng.uniform(-0.4, 0.4), 0.65)]
        elif scenario == "slalom":
            self.obstacles = [(x, y + rng.uniform(-0.22, 0.22), r) for x, y, r in
                              [(3.2, 3.7, 0.7), (5.8, 4.7, 0.8), (8.3, 3.45, 0.75)]]
        else:
            self.obstacles = []
            for _ in range(300):
                if len(self.obstacles) >= 9:
                    break
                candidate = (rng.uniform(2.3, 9.6), rng.uniform(1.0, 7.0), rng.uniform(0.3, 0.6))
                if all(math.hypot(candidate[0]-x, candidate[1]-y) > candidate[2]+r+0.65
                       for x, y, r in self.obstacles):
                    self.obstacles.append(candidate)
        self.steps, self.path_length, self.reward = 0, 0.0, 0.0
        self.status, self.last_action = "running", Action(0, 0)
        self.path = [(self.x, self.y)]
        self.min_clearance = self.clearance(self.x, self.y)
        return self.observe()

    def clearance(self, x, y):
        return min(x, y, self.width-x, self.height-y,
                   *(math.hypot(x-ox, y-oy)-r for ox, oy, r in self.obstacles)) - self.radius

    def raycast(self, relative_angle):
        a = self.heading + relative_angle
        dx, dy = math.cos(a), math.sin(a)
        hits = [self.ray_range]
        if abs(dx) > 1e-9:
            hits.append(((self.width if dx > 0 else 0)-self.x)/dx)
        if abs(dy) > 1e-9:
            hits.append(((self.height if dy > 0 else 0)-self.y)/dy)
        for ox, oy, radius in self.obstacles:
            px, py = self.x-ox, self.y-oy
            b = px*dx + py*dy
            disc = b*b - (px*px+py*py-radius*radius)
            if disc >= 0:
                t = -b-math.sqrt(disc)
                if t >= 0:
                    hits.append(t)
        return clip(min(hits), 0, self.ray_range)

    def observe(self):
        distance = math.hypot(self.goal[0]-self.x, self.goal[1]-self.y)
        return {"rays": [self.raycast(a) for a in self.ray_angles],
                "goal_distance": distance,
                "goal_angle": wrap(math.atan2(self.goal[1]-self.y, self.goal[0]-self.x)-self.heading),
                "linear": self.last_action.linear, "angular": self.last_action.angular}

    def step(self, action):
        if self.status != "running":
            raise RuntimeError("Episode is finished; reset before stepping")
        if not all(math.isfinite(v) for v in (action.linear, action.angular)):
            raise ValueError("Actions must be finite")
        action = action.bounded()
        before = self.observe()["goal_distance"]
        collision = False
        # Substeps prevent tunneling through small obstacles at the action limit.
        for _ in range(5):
            self.heading = wrap(self.heading + action.angular*self.dt/5)
            nx = self.x + action.linear*math.cos(self.heading)*self.dt/5
            ny = self.y + action.linear*math.sin(self.heading)*self.dt/5
            clearance = self.clearance(nx, ny)
            self.min_clearance = min(self.min_clearance, clearance)
            if clearance <= 0:
                collision = True
                break
            self.path_length += math.hypot(nx-self.x, ny-self.y)
            self.x, self.y = nx, ny
        self.steps += 1
        self.last_action = action
        self.path.append((self.x, self.y))
        obs = self.observe()
        success = obs["goal_distance"] < 0.4
        terminated = collision or success
        truncated = self.steps >= self.max_steps and not terminated
        self.status = "collision" if collision else "success" if success else "timeout" if truncated else "running"
        reward = 3*(before-obs["goal_distance"]) - 0.015 - 0.002*abs(action.angular)
        reward += 20 if success else -20 if collision else -3 if truncated else 0
        self.reward += reward
        return obs, reward, terminated, truncated, {"status": self.status}

    def state(self):
        return {"width": self.width, "height": self.height, "radius": self.radius,
                "x": self.x, "y": self.y, "heading": self.heading, "goal": self.goal,
                "obstacles": self.obstacles, "path": self.path[-1000:],
                "ray_angles": self.ray_angles, "observation": self.observe(),
                "steps": self.steps, "sim_seconds": round(self.steps*self.dt, 1),
                "status": self.status, "seed": self.seed, "scenario": self.scenario,
                "path_length": self.path_length, "min_clearance": self.min_clearance,
                "reward": self.reward}
