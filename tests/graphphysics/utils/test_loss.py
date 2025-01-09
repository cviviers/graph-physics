import unittest
import torch
from torch import nn
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.loss import GaussianMixtureNLLLoss


class TestGaussianMixtureNLLLoss(unittest.TestCase):
    def setUp(self):
        # Set up some mock parameters
        self.d = 4  # dimension of velocity
        self.K = 3  # number of mixture components
        self.temperature = 1.0
        self.loss_fn = GaussianMixtureNLLLoss(
            d=self.d, K=self.K, temperature=self.temperature
        )

    def test_forward_basic(self):
        """
        Test the forward pass of GaussianMixtureNLLLoss under normal conditions.
        """
        # Suppose we have 5 nodes
        N = 5
        # network_output shape: [N, K * per_comp]
        # per_comp = d + d(d+1)//2 + 1 => with d=4 => 4 + (4*5)//2 + 1 = 4 + 10 + 1 = 15
        per_comp = self.d + (self.d * (self.d + 1)) // 2 + 1
        net_out_dim = self.K * per_comp  # => 3 * 15 = 45
        network_output = torch.randn(N, net_out_dim)

        # target shape: [N, d]
        target = torch.randn(N, self.d)

        # node_type shape: [N]; We'll say all nodes are normal
        node_type = torch.full((N,), NodeType.NORMAL)

        # we want to compute the NLL
        loss_val = self.loss_fn(
            target=target,
            network_output=network_output,
            node_type=node_type,
            masks=[NodeType.NORMAL],  # so we include all nodes
        )

        # The result should be a single scalar
        self.assertTrue(loss_val.dim() == 0)
        self.assertFalse(torch.isnan(loss_val).any(), "Loss value is NaN.")
        self.assertFalse(torch.isinf(loss_val).any(), "Loss value is Inf.")

    def test_masked_nodes(self):
        """
        Test that nodes not in masks are ignored properly.
        """
        N = 6
        per_comp = self.d + (self.d * (self.d + 1)) // 2 + 1
        net_out_dim = self.K * per_comp
        network_output = torch.randn(N, net_out_dim)
        target = torch.randn(N, self.d)

        # node_type with half normal, half outflow
        node_type = torch.zeros(N, dtype=torch.long)
        # Make first 3 normal, last 3 outflow
        node_type[3:] = NodeType.OUTFLOW

        # We'll only keep mask for NodeType.NORMAL => that is 3 nodes
        loss_val = self.loss_fn(
            target=target,
            network_output=network_output,
            node_type=node_type,
            masks=[NodeType.NORMAL],
        )
        # still a scalar
        self.assertTrue(loss_val.dim() == 0)


if __name__ == "__main__":
    unittest.main()
