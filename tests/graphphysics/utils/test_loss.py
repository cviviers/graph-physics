import unittest
import torch
from torch import nn
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.loss import (
    GaussianMixtureNLLLoss,
    DiagonalGaussianMixtureNLLLoss,
)


class TestGaussianMixtureNLLLossDiagonal(unittest.TestCase):
    def setUp(self):
        # Suppose velocity dimension
        self.d = 3
        # Suppose we have K mixture components
        self.K = 2
        # We'll keep a temperature factor for scaling
        self.temperature = 1.0
        # Our diagonal GMM loss
        self.loss_fn = DiagonalGaussianMixtureNLLLoss(
            d=self.d, K=self.K, temperature=self.temperature
        )
        # The shape is 2d + 1 per component => 2*3 + 1 = 7
        # with K=2 => 2*7=14 output features per node

    def test_forward_basic(self):
        """
        Basic test: random N nodes, ensure we get a scalar loss and no NaNs/inf.
        """
        N = 5
        # GMM output shape: [N, K*(2*d + 1)] => [N, 14]
        network_output = torch.randn(N, self.K * (2 * self.d + 1))
        # target shape: [N, d] => [5,3]
        target = torch.randn(N, self.d)

        # node_type => let's say all are NORMAL
        node_type = torch.full((N,), NodeType.NORMAL)

        # run the loss
        loss_val = self.loss_fn(
            target=target,
            network_output=network_output,
            node_type=node_type,
            masks=[NodeType.NORMAL],  # we include these nodes
        )
        self.assertTrue(loss_val.dim() == 0, "Loss should be a scalar.")
        self.assertFalse(torch.isnan(loss_val).any(), "Loss returned NaN.")
        self.assertFalse(torch.isinf(loss_val).any(), "Loss returned Inf.")

    def test_masked_nodes(self):
        """
        Check that only masked node types are included in the GMM NLL.
        """
        N = 6
        net_out_dim = self.K * (2 * self.d + 1)
        network_output = torch.randn(N, net_out_dim)
        target = torch.randn(N, self.d)

        # half normal, half outflow
        node_type = torch.zeros(N, dtype=torch.long)  # Normal=0
        node_type[3:] = NodeType.OUTFLOW  # 1 => outflow

        # only compute for normal => first 3 nodes
        loss_val = self.loss_fn(
            target=target,
            network_output=network_output,
            node_type=node_type,
            masks=[NodeType.NORMAL],
        )
        self.assertTrue(loss_val.dim() == 0)


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
