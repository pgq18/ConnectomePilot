import copy
import unittest
import torch
from flylab.staged import freeze_learned_edges
from flylab.resume import load_learning_state
from tests.test_plasticity import policy


class StagedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(92)
        self.model = policy('ppo_edges')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.observations = torch.rand(3, 23)
        self.update(self.model, self.optimizer)

    def update(self, model, optimizer):
        mean, value, _, _ = model(self.observations, model.initial(3))
        loss = (mean - .2).square().mean() + (value - .3).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimizer.step()
        model.project()

    def test_freeze_preserves_predictions_and_readout_adam_then_only_readout_moves(self):
        before = copy.deepcopy(self.model.state_dict())
        moments = {name: copy.deepcopy(self.optimizer.state[p])
                   for name, p in self.model.named_parameters() if name != 'edge_weight'}
        with torch.no_grad(): expected = self.model(self.observations, self.model.initial(3))
        self.assertIsNotNone(self.model.edge_weight.grad)
        freeze_learned_edges(self.model, self.optimizer)
        self.assertFalse(self.model.edge_weight.requires_grad)
        self.assertIsNone(self.model.edge_weight.grad)
        self.assertNotIn(self.model.edge_weight, self.optimizer.state)
        self.assertFalse(any(p is self.model.edge_weight for g in self.optimizer.param_groups for p in g['params']))
        for name, p in self.model.named_parameters():
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            if name != 'edge_weight':
                for key, value in moments[name].items():
                    torch.testing.assert_close(self.optimizer.state[p][key], value, rtol=0, atol=0)
        actual = self.model(self.observations, self.model.initial(3))
        for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.update(self.model, self.optimizer)
        torch.testing.assert_close(self.model.edge_weight, before['edge_weight'], rtol=0, atol=0)
        self.assertIsNone(self.model.edge_weight.grad)
        self.assertFalse(torch.equal(self.model.actor.weight, before['actor.weight']))
        self.assertFalse(torch.equal(self.model.critic.weight, before['critic.weight']))

    def test_frozen_checkpoint_reloads_into_readout_mode_with_same_next_update(self):
        freeze_learned_edges(self.model, self.optimizer)
        checkpoint = copy.deepcopy({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()})
        restored = policy('readout')
        optimizer = torch.optim.Adam([p for p in restored.parameters() if p.requires_grad], lr=1.)
        load_learning_state(checkpoint, restored, optimizer)
        self.update(self.model, self.optimizer)
        self.update(restored, optimizer)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 3e-4)


if __name__ == '__main__': unittest.main()
