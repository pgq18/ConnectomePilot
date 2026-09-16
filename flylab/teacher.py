"""Privileged geometry teacher; it is never presented as a fly controller."""
from __future__ import annotations
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .controllers import base_action, compose
from .world import Action, wrap


def visible(a, b, obstacles, radius, margin=.12):
    a, b = np.asarray(a), np.asarray(b)
    delta = b - a
    for x, y, r in obstacles:
        t = np.clip(np.dot(np.array([x, y]) - a, delta) / max(1e-12, np.dot(delta, delta)), 0, 1)
        if np.linalg.norm(a + t * delta - [x, y]) < r + radius + margin:
            return False
    return True


class GeometryTeacher:
    def __init__(self, world):
        self.seed = world.seed
        self.route = self.plan(world)
        self.index = 1

    @staticmethod
    def plan(world):
        # Circumscribed polygons ensure segments between neighboring vertices
        # stay outside the inflated circles. Exact segment visibility is checked.
        count = 16
        nodes = [[world.x, world.y], list(world.goal)]
        for x, y, r in world.obstacles:
            rho = (r + world.radius + .24) / math.cos(math.pi / count)
            for angle in np.linspace(0, 2 * math.pi, count, endpoint=False):
                p = [x + rho * math.cos(angle), y + rho * math.sin(angle)]
                if world.clearance(*p) > .15:
                    nodes.append(p)
        nodes = np.asarray(nodes)
        a = nodes[:, None, :]
        delta = nodes[None, :, :] - a
        length2 = (delta ** 2).sum(-1)
        allowed = length2 > 0
        for x, y, r in world.obstacles:
            fraction = np.clip(((np.array([x, y]) - a) * delta).sum(-1) / np.maximum(length2, 1e-12), 0, 1)
            distance = np.linalg.norm(a + fraction[..., None] * delta - [x, y], axis=-1)
            threshold = np.full_like(distance, r + world.radius + .12)
            departure_margin = min(.12, max(0, world.clearance(world.x, world.y) - .005))
            threshold[0, :] = threshold[:, 0] = r + world.radius + departure_margin
            allowed &= distance >= threshold
        graph = csr_matrix(np.where(allowed, np.sqrt(length2), 0))
        distance, previous = dijkstra(graph, indices=0, return_predecessors=True)
        if not np.isfinite(distance[1]):
            raise RuntimeError(f'No teacher path in scene {world.seed}')
        indices, current = [1], 1
        while current != 0:
            current = int(previous[current]); indices.append(current)
        return nodes[indices[::-1]]

    def action(self, world):
        point = np.array([world.x, world.y])
        # Select the furthest visible remaining waypoint. This also supports
        # labels at student-visited states during DAgger.
        margin = min(.12, max(0, world.clearance(world.x, world.y) - .005))
        candidates = [j for j in range(self.index, len(self.route))
                      if visible(point, self.route[j], world.obstacles, world.radius, margin)]
        if candidates:
            self.index = max(candidates)
        else:
            self.route = self.plan(world); self.index = 1
        target = self.route[self.index]
        error = wrap(math.atan2(target[1] - world.y, target[0] - world.x) - world.heading)
        clearance_scale = np.clip(world.clearance(world.x, world.y) / .4, .25, 1)
        desired = Action(.8 * max(0, math.cos(error)) ** 2 * clearance_scale,
                         np.clip(2.5 * error, -2.2, 2.2))
        base = base_action(world.observe())
        normalized = np.clip([(desired.linear - base.linear) / .75,
                              (desired.angular - base.angular) / 1.8], -1, 1).astype(np.float32)
        return normalized


def apply_residual(world, normalized):
    return compose(base_action(world.observe()), Action(float(normalized[0]) * .75,
                                                       float(normalized[1]) * 1.8))[0]
