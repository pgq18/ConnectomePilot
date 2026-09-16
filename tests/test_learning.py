import unittest
import numpy as np
import torch
from scipy import sparse
from flylab.learning import LearningPolicy, FixedSparse, tensor_csr, LearningWorlds
from flylab.teacher import GeometryTeacher, apply_residual, visible
from flylab.world import NavigationWorld


class LearningTests(unittest.TestCase):
    def test_sparse_backward_matches_dense(self):
        matrix = sparse.csr_matrix(np.array([[0, .3, 0], [-.2, 0, .5], [.1, 0, 0]], dtype=np.float32))
        x = torch.randn(3, 4, requires_grad=True)
        y = FixedSparse.apply(x, tensor_csr(matrix), tensor_csr(matrix.T.tocsr()))
        y.square().sum().backward(); actual = x.grad.clone()
        reference = x.detach().clone().requires_grad_(True)
        (torch.tensor(matrix.toarray()) @ reference).square().sum().backward()
        torch.testing.assert_close(actual, reference.grad)

    def test_dynamic_gradient_and_reset(self):
        matrix = sparse.csr_matrix(np.array([[0, 0, .1], [.8, 0, 0], [0, .9, 0]], dtype=np.float32))
        model = LearningPolicy('fly', graph_override=(matrix, np.array([0]), np.array([1, 2])))
        x = torch.randn(6, 2, 23); starts = torch.zeros(6, 2, dtype=torch.bool); starts[0] = True
        means, _, hidden = model.sequence(x, model.initial(2), starts)
        means.square().sum().backward()
        for name in ('gain_raw', 'bias', 'leak_raw'):
            gradient = getattr(model, name).grad
            self.assertTrue(torch.isfinite(gradient).all()); self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(model.weight.grad)
        a = model(x[0], hidden, torch.ones(2, dtype=torch.bool))[0]
        b = model(x[0], model.initial(2))[0]
        torch.testing.assert_close(a, b)

    def test_teacher_respects_residual_interface(self):
        world = NavigationWorld(4100000, 'clutter'); teacher = GeometryTeacher(world)
        self.assertTrue(all(visible(a, b, world.obstacles, world.radius) for a, b in zip(teacher.route, teacher.route[1:])))
        action = teacher.action(world)
        self.assertTrue(np.all(np.abs(action) <= 1))
        final = apply_residual(world, action)
        self.assertTrue(0 <= final.linear <= 1.1 and abs(final.angular) <= 2.2)

    def test_training_maps_disjoint_and_reset(self):
        env = LearningWorlds(4, 61)
        self.assertTrue(all(10000000 <= w.seed < 11000000 for w in env.worlds))
        for w in env.worlds: w.steps = 599
        raw, _, done, _ = env.step(np.zeros((4, 2)))
        self.assertTrue(done.all()); np.testing.assert_array_equal(raw[:, 9:18], 0)
        self.assertTrue(all(10000000 <= w.seed < 11000000 for w in env.worlds))


if __name__ == '__main__': unittest.main()
