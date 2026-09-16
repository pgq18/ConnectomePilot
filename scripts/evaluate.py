"""Matched, seeded controller evaluations. No synthetic fly fallback."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.world import NavigationWorld
from flylab.controllers import Controller
from flylab.brain import FlyBrain, DATA


def evaluate(mode, seeds, scenario, brain=None):
    rows = []
    controller = Controller(mode, brain)
    for seed in seeds:
        world = NavigationWorld(seed, scenario)
        controller.reset(seed)
        start = time.perf_counter()
        while world.status == "running":
            action, _ = controller.act(world.observe())
            world.step(action)
        rows.append({"mode": mode, "seed": seed, "scenario": scenario,
                     "status": world.status, "steps": world.steps,
                     "path_length": world.path_length, "min_clearance": world.min_clearance,
                     "reward": world.reward, "wall_seconds": time.perf_counter()-start})
        print(f"{mode} / seed {seed}: {world.status}, {world.steps} steps, "
              f"{rows[-1]['wall_seconds']:.1f} s", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", default=["base", "reflex"], choices=["base", "reflex", "fly"])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--scenario", default="slalom", choices=["slalom", "single", "clutter"])
    p.add_argument("--output", type=Path, default=ROOT/"results"/"evaluation.json")
    a = p.parse_args()
    if not 1 <= a.episodes <= 10000:
        p.error("episodes must be between 1 and 10000")
    brain = FlyBrain() if "fly" in a.modes else None
    rows = []
    for mode in a.modes:
        rows += evaluate(mode, range(a.seed_start, a.seed_start+a.episodes), a.scenario, brain)
    summary = {mode: {"episodes": a.episodes,
                      "successes": sum(r["status"]=="success" for r in rows if r["mode"]==mode),
                      "collisions": sum(r["status"]=="collision" for r in rows if r["mode"]==mode),
                      "timeouts": sum(r["status"]=="timeout" for r in rows if r["mode"]==mode)}
               for mode in a.modes}
    payload = {"scenario": a.scenario, "seed_start": a.seed_start, "summary": summary, "episodes": rows,
               "interpretation": "Matched simulation runs; does not validate biology or hardware transfer.",
               "brain_used": "fly" in a.modes, "brain_weights_trained": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
