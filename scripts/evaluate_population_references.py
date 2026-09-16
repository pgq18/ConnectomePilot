"""Untrained engineering references on the population trial's held-out maps."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flylab.controllers import reflex_residual
from flylab.population_ppo import WorldBatch
from train_population import summarize, write_json


def main():
    results = []
    for kind in ('base', 'reflex'):
        for condition in ('clean', 'noise', 'delay'):
            rows = []
            # No padding is needed for these CPU-only policies.
            worlds = WorldBatch(100, 0, 'clutter', list(range(3000000, 3000100)), condition)
            raw = worlds.observe()
            finished = np.zeros(100, dtype=bool)
            for _ in range(600):
                actions = np.zeros((100, 2), dtype=np.float32)
                if kind == 'reflex':
                    for i, features in enumerate(raw):
                        # Use the same perturbed/delayed range observation as PPO.
                        residual = reflex_residual({'rays': (features[:9] * 3).tolist()})
                        actions[i] = np.clip([residual.linear / .75, residual.angular / 1.8], -1, 1)
                raw, _, done, episodes = worlds.step(actions, autoreset=False)
                for episode in episodes:
                    episode.pop('slot')
                    rows.append(episode)
                finished |= done
                if finished.all():
                    break
            assert len(rows) == 100
            result = {'kind': kind, 'condition': condition, 'summary': summarize(rows), 'episodes': rows}
            results.append(result)
            print(kind, condition, result['summary'], flush=True)
    write_json(ROOT / 'results/population/references.json', results)


if __name__ == '__main__':
    main()
