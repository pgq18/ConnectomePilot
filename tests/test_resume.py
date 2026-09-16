from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from flylab.plasticity import PlasticityWorlds, ThreeFactor
from flylab.resume import environment_state, restore_environment, load_learning_state, save_checkpoint
from tests.test_plasticity import policy


class ResumeTests(unittest.TestCase):
    def test_environment_round_trip_and_future_reset(self):
        original = PlasticityWorlds(2,1000071)
        for _ in range(3): original.step(np.array([[.1,.2],[-.2,-.1]]))
        original.worlds[0].steps = 599
        restored = PlasticityWorlds(2,99)
        restore_environment(restored, environment_state(original))
        for _ in range(3):
            a = original.step(np.array([[.1,.2],[-.2,-.1]]))
            b = restored.step(np.array([[.1,.2],[-.2,-.1]]))
            for x,y in zip(a[:3],b[:3]): np.testing.assert_array_equal(x,y)
            self.assertEqual(a[3],b[3])
        self.assertEqual(environment_state(original),environment_state(restored))

    def test_checkpoint_restores_adam_and_next_update(self):
        torch.manual_seed(3); first = policy('ppo_edges')
        optimizer = torch.optim.Adam([p for p in first.parameters() if p.requires_grad],lr=3e-4)
        observations = torch.rand(2,23)
        def update(model,opt):
            loss=model(observations,model.initial(2))[0].square().sum()
            opt.zero_grad();loss.backward();opt.step()
        update(first,optimizer)
        env = PlasticityWorlds(2,71)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'resume.pt'
            save_checkpoint(path,first,optimizer,None,{},66560,env,env.observe(),first.initial(2),torch.ones(2,dtype=torch.bool))
            saved=torch.load(path,weights_only=True)
        second=policy('ppo_edges')
        optimizer2=torch.optim.Adam([p for p in second.parameters() if p.requires_grad],lr=1.)
        load_learning_state(saved,second,optimizer2)
        self.assertEqual(optimizer2.param_groups[0]['lr'],3e-4)
        update(first,optimizer);update(second,optimizer2)
        for a,b in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.set_rng_state(saved['rollout_state']['torch_cpu_rng']);a=torch.rand(4)
        torch.set_rng_state(saved['rollout_state']['torch_cpu_rng']);torch.testing.assert_close(a,torch.rand(4))

    def test_local_weights_resume_with_episode_traces_reset(self):
        first=policy('three_factor');local=ThreeFactor(first,2)
        local.edge_trace.fill_(.3);local.actor_trace.fill_(.2);local.bias_trace.fill_(.1)
        with torch.no_grad():first.edge_weight.add_(.01)
        checkpoint={'model':first.state_dict(),'local_traces':{'edge':local.edge_trace,'actor':local.actor_trace,'bias':local.bias_trace}}
        second=policy('three_factor');local2=ThreeFactor(second,2)
        load_learning_state(checkpoint,second,local=local2)
        torch.testing.assert_close(second.edge_weight,first.edge_weight)
        self.assertEqual(float(local2.edge_trace.abs().sum()),0)
        load_learning_state(checkpoint,second,local=local2,restore_traces=True)
        torch.testing.assert_close(local.edge_trace,local2.edge_trace)


if __name__=='__main__':unittest.main()
