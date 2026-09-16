"""Train a residual readout, save checkpoints, and evaluate disjoint scenes."""
import argparse
import hashlib
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def save_policy(model, path):
    temporary = path.with_name(path.stem+'.pending.zip')
    model.save(temporary)
    temporary.replace(path)


def brain_digest(brain):
    if brain is None:
        return None
    digest = hashlib.sha256()
    for array in (brain.weights.indptr, brain.weights.indices, brain.weights.data):
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def summarize(rows):
    return {mode: {'episodes': len(group),
                   'successes': sum(r['status'] == 'success' for r in group),
                   'collisions': sum(r['status'] == 'collision' for r in group),
                   'timeouts': sum(r['status'] == 'timeout' for r in group),
                   'mean_task_return': sum(r['task_return'] for r in group)/len(group)}
            for mode in dict.fromkeys(r['mode'] for r in rows)
            for group in [[r for r in rows if r['mode'] == mode]]}


def evaluate(model, initial, env, seeds, output, config):
    from flylab.controllers import Controller
    from flylab.world import NavigationWorld
    rows = []
    modes = ['base', 'reflex'] + (['fly_hand_mapping'] if env.brain else []) + ['ppo_initial', 'ppo_trained']
    for mode in modes:
        for seed in seeds:
            start = time.perf_counter()
            if mode.startswith('ppo_'):
                obs, _ = env.reset(options={'scene_seed': seed})
                policy = model if mode == 'ppo_trained' else initial
                done = False
                while not done:
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                world = env.world
            else:
                world = NavigationWorld(seed, env.scenario)
                controller = Controller('fly' if mode == 'fly_hand_mapping' else mode, env.brain)
                controller.reset(seed)
                while world.status == 'running':
                    action, _ = controller.act(world.observe())
                    world.step(action)
            row = {'mode': mode, 'scene_seed': seed, 'scenario': env.scenario,
                   'status': world.status, 'steps': world.steps, 'task_return': world.reward,
                   'path_length': world.path_length, 'min_clearance': world.min_clearance,
                   'wall_seconds': time.perf_counter()-start}
            rows.append(row)
            write_json(output/'evaluation.json', {'complete': False, 'config': config,
                       'summary': summarize(rows), 'episodes': rows})
            print(f"Eval {mode} / {seed}: {world.status}, {world.steps} steps", flush=True)
    result = {'complete': True, 'config': {**config, 'status': 'complete'}, 'summary': summarize(rows), 'episodes': rows,
              'notes': 'All modes share held-out scenes. Task return excludes the PPO-only residual regularizer. Different features prevent attributing differences solely to connectivity.'}
    write_json(output/'evaluation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', choices=['rays', 'fly'], default='rays')
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scenario', choices=['single', 'slalom', 'clutter'], default='clutter')
    parser.add_argument('--brain-backend', choices=['scipy', 'active'], default='active')
    parser.add_argument('--rollout-steps', type=int, default=1024)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--eval-seed-start', type=int, default=2000000)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--memory-limit-mib', type=float, default=1536)
    parser.add_argument('--max-train-seconds', type=float, default=600)
    args = parser.parse_args()
    if args.steps < 1 or args.rollout_steps < 64 or args.rollout_steps % 64:
        parser.error('Positive steps and rollout-steps divisible by 64 are required')
    if not 1 <= args.eval_episodes <= 1000 or args.eval_seed_start < 1000000:
        parser.error('Evaluation requires 1..1000 scenes with seeds >= 1000000')
    if args.memory_limit_mib <= 0 or args.max_train_seconds <= 0:
        parser.error('Resource budgets must be positive')
    output = args.output or ROOT/'results'/'ppo'/args.features
    if (output/'run.json').exists():
        parser.error('Output already has a run. Choose a new --output directory to preserve it.')
    output.mkdir(parents=True, exist_ok=True)
    import torch
    import numpy as np
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from flylab.gym_env import ResidualNavigationEnv
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    def peak_mib():
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw/(1024**2 if sys.platform == 'darwin' else 1024)
    core = ResidualNavigationEnv(features=args.features, scenario=args.scenario, brain_backend=args.brain_backend)
    core.reset(seed=args.seed)  # Includes one-time JIT compilation before timing.
    frozen_before = brain_digest(core.brain)
    env = Monitor(core, str(output/'train'))
    if args.resume:
        model = PPO.load(args.resume, env=env, device='cpu')
        model.set_random_seed(args.seed)
    else:
        model = PPO('MlpPolicy', env, seed=args.seed, device='cpu', verbose=0,
                    n_steps=args.rollout_steps, batch_size=64, learning_rate=3e-4,
                    policy_kwargs={'net_arch': [64, 64]})
    model.set_logger(configure(str(output), ['csv']))
    start_steps = model.num_timesteps
    start = time.perf_counter()
    config = {'status': 'training', 'features': args.features, 'scenario': args.scenario,
              'seed': args.seed, 'training_steps_requested': args.steps,
              'initial_timesteps': start_steps, 'rollout_steps': model.n_steps,
              'batch_size': model.batch_size, 'learning_rate': 3e-4,
              'train_scene_seed_range': [0, 999999],
              'eval_scene_seeds': list(range(args.eval_seed_start, args.eval_seed_start+args.eval_episodes)),
              'connectome_trainable': False, 'encoder_trainable': False,
              'trainable_parameters': sum(p.numel() for p in model.policy.parameters() if p.requires_grad),
              'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'parallel_environments': 1,
              'memory_limit_mib': args.memory_limit_mib, 'max_train_seconds': args.max_train_seconds,
              'brain_backend': args.brain_backend if core.brain else None,
              'brain_hash_before': frozen_before,
              'versions': {'torch': torch.__version__, 'stable_baselines3': sb3.__version__, 'numpy': np.__version__},
              'resume_from': str(args.resume) if args.resume else None,
              'started_at': datetime.now(timezone.utc).isoformat()}
    write_json(output/'run.json', config)
    save_policy(model, output/'initial_policy.zip')

    class Progress(BaseCallback):
        stop_reason = None

        def _on_step(self):
            if self.n_calls % 256 == 0:
                steps = self.num_timesteps-start_steps
                elapsed = time.perf_counter()-start
                write_json(output/'progress.json', {'phase': 'training', 'steps': steps,
                           'requested_steps': args.steps, 'elapsed_seconds': elapsed,
                           'steps_per_second': steps/max(elapsed, .001), 'peak_process_rss_mib': peak_mib()})
                if self.n_calls % 1024 == 0:
                    print(f"PPO {args.features}: {steps}/{args.steps} steps, {steps/elapsed:.1f} steps/s", flush=True)
                if peak_mib() > args.memory_limit_mib:
                    self.stop_reason = 'memory_budget'
                    return False
                if elapsed > args.max_train_seconds:
                    self.stop_reason = 'time_budget'
                    return False
            return True

        def _on_rollout_start(self):
            # A rollout starts after the previous PPO update has completed.
            if self.model.num_timesteps > start_steps:
                save_policy(self.model, output/'checkpoint.zip')

    try:
        progress = Progress()
        model.learn(total_timesteps=args.steps, callback=progress, reset_num_timesteps=not bool(args.resume))
        if progress.stop_reason == 'memory_budget':
            save_policy(model, output/'resource_stopped_policy.zip')
            config.update(status='resource_stopped', stop_reason=progress.stop_reason,
                          total_timesteps=model.num_timesteps, peak_process_rss_mib=peak_mib())
            write_json(output/'run.json', config)
            write_json(output/'progress.json', config)
            print('Memory budget reached; checkpoint saved and training stopped.', flush=True)
            return
        save_policy(model, output/'policy.zip')
        config.update(status='evaluating', training_steps_actual=model.num_timesteps-start_steps,
                      total_timesteps=model.num_timesteps, training_seconds=time.perf_counter()-start,
                      brain_hash_after=brain_digest(core.brain), stop_reason=progress.stop_reason,
                      peak_process_rss_mib=peak_mib())
        if config['brain_hash_after'] != frozen_before:
            raise RuntimeError('Frozen connectome changed during training')
        initial_state = PPO.load(output/'initial_policy.zip', device='cpu')
        delta = sum(float(torch.sum((a-b)**2)) for a, b in zip(
            model.policy.state_dict().values(), initial_state.policy.state_dict().values()))**.5
        config['policy_parameter_delta_l2'] = delta
        write_json(output/'run.json', config)
        write_json(output/'progress.json', {'phase': 'evaluating', 'steps': config['training_steps_actual']})
        result = evaluate(model, initial_state, core, config['eval_scene_seeds'], output, config.copy())
        config['status'] = 'complete'
        write_json(output/'run.json', config)
        write_json(output/'progress.json', {'phase': 'complete', 'steps': config['training_steps_actual']})
        print(json.dumps(result['summary'], indent=2), flush=True)
        print(f"Saved policy and evaluation: {output}", flush=True)
    except KeyboardInterrupt:
        save_policy(model, output/'interrupted_policy.zip')
        config.update(status='interrupted', total_timesteps=model.num_timesteps)
        write_json(output/'run.json', config)
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
