"""Every action is decomposed into a base action and an explicit bounded residual."""
from .world import Action, clip
import math


def base_action(observation):
    a = observation["goal_angle"]
    return Action(0.85 * max(0.12, math.cos(a)), clip(1.8*a, -1.5, 1.5))


def reflex_residual(observation):
    """An engineered range-sensor baseline, never labeled as a fly brain."""
    rays = observation["rays"]
    proximity = [max(0, 1-r/2.0) for r in rays]
    left, right = sum(proximity[5:]), sum(proximity[:4])
    turn = 1.6*(right-left)
    if abs(turn) < 0.08 and proximity[4] > 0.15:
        turn = -0.9
    slow = -0.65 * max(proximity[2:7])
    return Action(slow, turn)


def compose(base, residual, gain=1.0):
    residual = Action(clip(residual.linear*gain, -0.75, 0.25),
                      clip(residual.angular*gain, -1.8, 1.8))
    return Action(base.linear+residual.linear, base.angular+residual.angular).bounded(), residual


class Controller:
    def __init__(self, mode="reflex", brain=None, gain=1.0):
        if mode not in ("base", "reflex", "fly"):
            raise ValueError("Unknown controller")
        if mode == "fly" and brain is None:
            raise ValueError("Fly controller requires measured connectome data")
        self.mode, self.brain, self.gain = mode, brain, gain

    def reset(self, seed):
        if self.brain is not None and self.mode == "fly":
            self.brain.reset(seed)

    def act(self, observation):
        base = base_action(observation)
        features = {}
        if self.mode == "base":
            residual = Action(0, 0)
        elif self.mode == "reflex":
            residual = reflex_residual(observation)
        else:
            residual, features = self.brain.act(observation)
        action, residual = compose(base, residual, self.gain)
        return action, {"base": [base.linear, base.angular],
                        "residual": [residual.linear, residual.angular],
                        "final": [action.linear, action.angular], "neural": features}
