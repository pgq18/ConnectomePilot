from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from flylab.plasticity import PlasticityWorlds
from flylab.resume import save_checkpoint, load_learning_state, environment_state
from flylab.continuation import restore_rollout_state
from tests.test_plasticity import policy


class ContinuationTests(unittest.TestCase):
    def test_full_frozen_rollout_restores_actions_episode_reset_and_next_update(self):
        torch.manual_seed(174)
        first = policy('readout')
        optimizer = torch.optim.Adam([p for p in first.parameters() if p.requires_grad], lr=3e-4)
        env = PlasticityWorlds(2, 1000071)
        raw = env.observe(); hidden = first.initial(2); starts = torch.ones(2,dtype=torch.bool)

        def advance(model, opt, world, obs, state, begins):
            actions=[]; episodes=[]; rewards=[]
            for _ in range(3):
                obs_tensor=torch.as_tensor(obs)
                mean, value, next_state, _=model(obs_tensor,state,begins)
                latent=mean.detach()+model.std*torch.randn_like(mean)
                action=latent.tanh().numpy()
                obs,reward,done,rows=world.step(action)
                loss=(mean-latent).square().mean()+(value-torch.as_tensor(reward)).square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();model.project()
                state=next_state.detach();begins=torch.as_tensor(done)
                actions.append(action.copy());rewards.append(reward.copy());episodes.extend(rows)
            return obs,state,begins,actions,rewards,episodes

        raw,hidden,starts,*_=advance(first,optimizer,env,raw,hidden,starts)
        env.worlds[0].steps=599
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'full.pt'
            save_checkpoint(path,first,optimizer,None,{'mode':'readout'},262144,env,raw,hidden,starts)
            checkpoint=torch.load(path,weights_only=True)
        expected=advance(first,optimizer,env,raw,hidden,starts)
        restored=policy('readout')
        restored_opt=torch.optim.Adam([p for p in restored.parameters() if p.requires_grad],lr=1.)
        restored_env=PlasticityWorlds(2,999)
        load_learning_state(checkpoint,restored,restored_opt)
        new_raw,new_hidden,new_starts=restore_rollout_state(checkpoint,restored_env,'cpu')
        actual=advance(restored,restored_opt,restored_env,new_raw,new_hidden,new_starts)
        np.testing.assert_array_equal(expected[0],actual[0])
        for a,b in zip(expected[1:3],actual[1:3]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(expected[3:5],actual[3:5]):np.testing.assert_array_equal(a,b)
        self.assertEqual(expected[5],actual[5]);self.assertTrue(actual[5])
        self.assertEqual(environment_state(env),environment_state(restored_env))
        for name,value in first.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
        self.assertFalse(restored.edge_weight.requires_grad)
        self.assertIsNone(restored.edge_weight.grad)


if __name__=='__main__':unittest.main()
