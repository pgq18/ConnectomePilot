"""Validate the teacher and collect disjoint demonstrations for learning."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flylab.teacher import GeometryTeacher, apply_residual
from flylab.population import observation_vector
from flylab.world import NavigationWorld
from scripts.train_population import write_json, summarize


def episode(seed, perturb=False):
    world = NavigationWorld(seed, 'clutter')
    teacher = GeometryTeacher(world)
    rng = np.random.default_rng(seed)
    previous = None
    observations, actions = [], []
    jitter = 0.
    for t in range(600):
        raw = world.observe()
        observations.append(observation_vector(raw, previous)); previous = raw['rays']
        label = teacher.action(world); actions.append(label)
        executed = label
        if perturb and t % 70 < 12:
            executed = np.clip(label + rng.normal(0, .18, 2), -1, 1)
        action = apply_residual(world, executed)
        jitter += abs(action.angular - world.last_action.angular)
        world.step(action)
        if world.status != 'running': break
    row = {'scene_seed': seed, 'status': world.status, 'steps': world.steps,
           'task_return': world.reward, 'path_length': world.path_length, 'min_clearance': world.min_clearance,
           'mean_angular_action_change': jitter / world.steps}
    return np.asarray(observations), np.asarray(actions), row


def main():
    p = argparse.ArgumentParser(); p.add_argument('--episodes', type=int, default=128)
    p.add_argument('--check-only', action='store_true'); args = p.parse_args()
    root = ROOT / 'results/learning'; root.mkdir(parents=True, exist_ok=True)
    checks = [episode(seed)[2] for seed in range(4100000, 4100050)]
    print('teacher development', summarize(checks), flush=True)
    write_json(root / 'teacher-development.json', {'summary': summarize(checks), 'episodes': checks})
    if args.check_only: return
    for split, seeds in [('train', range(10000000, 10000000 + args.episodes)),
                         ('validation', range(4200000, 4200032))]:
        obs, labels, offsets, rows = [], [], [0], []
        for seed in seeds:
            x, y, row = episode(seed, perturb=split == 'train')
            obs.append(x); labels.append(y); offsets.append(offsets[-1] + len(x)); rows.append(row)
        np.savez_compressed(root / f'{split}.npz', observations=np.concatenate(obs), actions=np.concatenate(labels),
                            offsets=np.asarray(offsets), scene_seeds=np.asarray(list(seeds)))
        write_json(root / f'{split}-collection.json', {'summary': summarize(rows), 'episodes': rows})
        print(split, offsets[-1], summarize(rows), flush=True)


if __name__ == '__main__': main()
