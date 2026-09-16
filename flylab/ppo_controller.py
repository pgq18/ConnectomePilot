"""Load this project's trained residual policy for the browser playground."""
import json
from pathlib import Path

from .brain import DATA, FlyBrain
from .controllers import base_action, compose
from .world import Action

POLICY_ROOT = Path(__file__).resolve().parents[1]/'results'/'ppo'


def policy_status():
    result = {}
    for features in ('rays', 'fly'):
        folder = POLICY_ROOT/features
        try:
            run = json.loads((folder/'run.json').read_text())
            ready = (folder/'policy.zip').is_file() and run['features'] == features and run['status'] == 'complete'
        except (OSError, ValueError, KeyError):
            run, ready = {}, False
        result['ppo-'+features] = {'available': ready,
                                  'training_steps': run.get('training_steps_actual', 0),
                                  'scenario': run.get('scenario'), 'status': run.get('status', 'missing')}
    return result


class PPOController:
    def __init__(self, mode, brain=None, gain=1.0):
        if mode not in ('ppo-rays', 'ppo-fly') or not policy_status()[mode]['available']:
            raise ValueError('该 PPO 策略尚未完成训练与评估')
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        self.mode, self.gain = mode, gain
        self.features = mode.removeprefix('ppo-')
        folder = POLICY_ROOT/self.features
        self.run = json.loads((folder/'run.json').read_text())
        self.model = PPO.load(folder/'policy.zip', device='cpu')
        self.brain = None
        if self.features == 'fly':
            self.brain = brain or FlyBrain(DATA, backend=self.run.get('brain_backend', 'scipy'))

    def reset(self, seed):
        if self.brain is not None:
            self.brain.reset(seed)

    def act(self, raw):
        from .gym_env import policy_observation
        observation, neural = policy_observation(raw, self.brain)
        normalized, _ = self.model.predict(observation, deterministic=True)
        base = base_action(raw)
        final, residual = compose(base, Action(float(normalized[0])*.75,
                                               float(normalized[1])*1.8), self.gain)
        return final, {'base': [base.linear, base.angular],
                       'residual': [residual.linear, residual.angular],
                       'final': [final.linear, final.angular], 'neural': neural,
                       'policy_trained': True, 'training_steps': self.run['training_steps_actual'],
                       'online_learning': False}
