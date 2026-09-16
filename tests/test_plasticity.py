import unittest
import numpy as np
import torch
from flylab.plasticity import PlasticPolicy, ThreeFactor, PlasticityWorlds


def policy(mode):
    return PlasticPolicy(mode, {'source': [0,1,2], 'target': [1,2,0], 'weight': [.8,-.4,.1],
        'inputs': [0], 'channels': [0], 'descending': [1,2], 'global_indices': [0,1,2]})


class PlasticityTests(unittest.TestCase):
    def test_projection_changes_base_and_preserves_signs(self):
        m = policy('ppo_edges')
        with torch.no_grad(): m.edge_weight.copy_(torch.tensor([-2.,-4.,3.]))
        m.project()
        torch.testing.assert_close(m.edge_weight, torch.tensor([0.,-1.,1.]))

    def test_edge_gradients_and_reset(self):
        m = policy('ppo_edges'); obs = torch.rand(4,23)
        mean, _, hidden, _ = m(obs, m.initial(4))
        mean.square().sum().backward()
        self.assertGreater(float(m.edge_weight.grad.abs().sum()), 0)
        a = m(obs, hidden, torch.ones(4,dtype=torch.bool))[0]
        b = m(obs, m.initial(4))[0]
        torch.testing.assert_close(a,b)
        fixed = policy('readout')
        self.assertFalse(fixed.edge_weight.requires_grad)

    def test_gaussian_score_matches_autograd(self):
        m = policy('three_factor'); mean = torch.tensor([[.1,-.2]], requires_grad=True)
        latent = torch.tensor([[.3,-.1]])
        m.log_prob(mean, latent).sum().backward()
        torch.testing.assert_close(mean.grad, (latent-mean.detach())/m.std**2)

    def test_sparse_message_gradient_matches_dense_reference(self):
        m = policy('ppo_edges'); obs = torch.rand(2,23)
        actual = m(obs,m.initial(2))[2]
        actual.square().sum().backward(); actual_grad = m.edge_weight.grad.clone()
        weights = m.edge_weight.detach().clone().requires_grad_(True)
        matrix = torch.zeros(3,3).index_put((m.target,m.source),weights)
        hidden = torch.zeros(2,3); external = torch.zeros_like(hidden)
        external[:,0] = .9*(1-obs[:,0])
        for _ in range(5): hidden = torch.tanh(.5*hidden+hidden@matrix.T+external)
        hidden.square().sum().backward()
        torch.testing.assert_close(actual,hidden)
        torch.testing.assert_close(actual_grad,weights.grad)

    def test_reward_gating_and_episode_local_traces(self):
        m = policy('three_factor'); learner = ThreeFactor(m, 2)
        hidden = torch.tensor([[.4,.2,-.3],[.1,.5,-.2]])
        features = torch.ones(2,7)*.2
        mean = torch.zeros(2,2); zeros = torch.zeros(2); latent = torch.ones(2,2)*.1
        before = m.edge_weight.clone()
        learner.update(hidden,features,mean,zeros,latent,zeros,zeros,torch.zeros(2,dtype=torch.bool))
        torch.testing.assert_close(before,m.edge_weight)
        self.assertGreater(float(learner.edge_trace.abs().sum()),0)
        learner.update(hidden,features,mean,zeros,latent,torch.ones(2),zeros,torch.tensor([True,False]))
        self.assertGreater(float((before-m.edge_weight).abs().sum()),0)
        self.assertEqual(float(learner.edge_trace[0].abs().sum()),0)
        self.assertGreater(float(learner.edge_trace[1].abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_training_maps_and_first_observation(self):
        env = PlasticityWorlds(4,71)
        self.assertTrue(all(20000000 <= w.seed < 21000000 for w in env.worlds))
        for w in env.worlds: w.steps = 599
        obs,_,done,_ = env.step(np.zeros((4,2)))
        self.assertTrue(done.all()); np.testing.assert_array_equal(obs[:,9:18],0)
        self.assertTrue(all(20000000 <= w.seed < 21000000 for w in env.worlds))


if __name__ == '__main__': unittest.main()
