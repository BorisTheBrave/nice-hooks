import nice_hooks
import unittest
import torch as t
import torch.nn as nn



def init(module: nn.Module):
    with t.no_grad():
        for mod in module.modules():
            if isinstance(mod, nn.Linear):
                t.fill_(mod.weight, 1)
                if mod.bias is not None:
                    t.zero_(mod.bias)

model1 = nn.Sequential(
    nn.Linear(1, 10),
    nn.Linear(10, 1),
)
init(model1)


model2 = nn.Sequential(
    nn.Linear(1, 10),
    nn.Sequential(
        nn.Linear(10, 1),
        nn.Linear(1, 1),
    )
)
init(model2)


class TestNiceHooks(unittest.TestCase):
    def test_return_all_activations(self):
        r, a = nice_hooks.run(model1, t.zeros(1), return_activations=True)
        self.assertEqual(["0", "1", ""], list(a))

    def test_return_single_activation(self):
        r, a = nice_hooks.run(model1, t.zeros(1), return_activations=["1"])
        self.assertEqual(["1"], list(a))

    def test_return_wildcard_activation(self):
        r, a = nice_hooks.run(model2, t.zeros(1), return_activations=["1.*"])
        self.assertEqual(["1.0", "1.1"], list(a))

    def test_return_slice_activation(self):
        r, a = nice_hooks.run(model2, t.zeros(1), return_activations=["0[0:5]"])
        self.assertEqual(["0[0:5]"], list(a))
        self.assertEqual((5,), a["0[0:5]"].shape)

    def tests_with_activation1(self):
        r = nice_hooks.run(model1, t.zeros(1), with_activations={"1": t.ones(1)})
        self.assertEqual(t.ones(1), r)

    def tests_with_activation2(self):
        r = nice_hooks.run(model1, t.zeros(1), with_activations={"0": t.ones(10)})
        self.assertEqual(t.ones(1) * 10, r)

if __name__ == '__main__':
    unittest.main()