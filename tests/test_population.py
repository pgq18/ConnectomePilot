import unittest
import numpy as np
import torch
from flylab.population import observation_vector,input_map
from flylab.population_ppo import Policy,WorldBatch,advantages


class PopulationContractTests(unittest.TestCase):
    def test_approach_reset_and_direction_preservation(self):
        raw={'rays':[3.]*9,'goal_angle':0.,'goal_distance':8.,'linear':0.,'angular':0.}
        raw['rays'][0]=.3
        x=observation_vector(raw)
        self.assertEqual(x.shape,(23,))
        self.assertAlmostEqual(float(x[0]),.1)
        np.testing.assert_array_equal(x[9:18],0)
        y=observation_vector(raw,[3.]*9)
        self.assertEqual(y[9],1)
        self.assertEqual(y[17],0)

    def test_input_channels_disjoint_and_complete(self):
        ids=np.arange(80)
        idx,ch=input_map(np.array(['LC4']*80),np.array(['L']*40+['R']*40),ids)
        self.assertEqual(len(set(idx)),len(idx))
        self.assertEqual(set(ch),set(range(18)))

    def test_gae_stops_at_episode_boundary(self):
        rewards=torch.tensor([[1.],[2.],[50.]])
        values=torch.zeros_like(rewards)
        done=torch.tensor([[False],[True],[False]])
        adv,returns=advantages(rewards,values,done,torch.tensor([10.]),gamma=1,lam=1)
        torch.testing.assert_close(adv,torch.tensor([[3.],[2.],[60.]]))

    def test_gru_sequence_matches_online_and_resets(self):
        torch.set_num_threads(1); torch.manual_seed(9)
        policy=Policy('gru',23)
        x=torch.randn(7,3,23)
        h=torch.randn(3,policy.width)
        starts=torch.zeros(7,3,dtype=torch.bool); starts[3,1]=True
        mean,value=policy.sequence(x,h,starts)
        outputs=[]; state=h
        for obs,reset in zip(x,starts):
            mu,_,state=policy(obs,state,reset); outputs.append(mu)
        torch.testing.assert_close(mean,torch.stack(outputs))
        independent,_,_=policy(x[3,1:2],torch.zeros_like(h[1:2]))
        torch.testing.assert_close(mean[3,1:2],independent)
        loss=mean.sum()+value.sum(); loss.backward()
        self.assertGreater(float(policy.core.weight_hh.grad.abs().sum()),0)

    def test_training_maps_and_terminal_reset(self):
        batch=WorldBatch(4,11)
        for w in batch.worlds: self.assertLess(w.seed,1000000)
        batch.worlds[0].steps=599
        _,_,done,episodes=batch.step(np.zeros((4,2),np.float32))
        self.assertTrue(done[0])
        self.assertEqual(batch.worlds[0].steps,0)
        self.assertEqual(episodes[0]['status'],'timeout')
        np.testing.assert_array_equal(observation_vector(batch.worlds[0].observe(),batch.previous[0])[9:18],0)

if __name__=='__main__': unittest.main()
