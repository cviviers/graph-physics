import unittest
import torch
from graphphysics.models.simulator import sample_gmm


class TestGMMSampler(unittest.TestCase):
    def test_sample_gmm_shapes(self):
        """
        Test that sample_gmm returns the correct shape and no NaNs.
        """
        N = 5  # number of nodes
        d = 3  # dimension of velocity
        K = 2  # mixture components
        temperature = 1.0

        # per_comp = d + d(d+1)//2 + 1
        per_comp = d + (d * (d + 1)) // 2 + 1
        out_dim = K * per_comp  # total GMM param dimension per node

        # create random GMM parameters
        network_output = torch.randn(N, out_dim)

        velocities = sample_gmm(network_output, d=d, K=K, temperature=temperature)
        # shape should be [N, d]
        self.assertEqual(velocities.shape, (N, d))
        self.assertFalse(torch.isnan(velocities).any(), "Sampler returned NaN.")
        self.assertFalse(torch.isinf(velocities).any(), "Sampler returned Inf.")

    def test_sample_gmm_temperature(self):
        """
        Check that different temperatures produce different magnitude samples.
        """
        N = 4
        d = 2
        K = 2
        per_comp = d + (d * (d + 1)) // 2 + 1
        out_dim = K * per_comp
        network_output = torch.randn(N, out_dim)

        # sample with T=0.1
        velocities_lowT = sample_gmm(network_output, d, K, temperature=0.1)
        # sample with T=5.0
        velocities_highT = sample_gmm(network_output, d, K, temperature=5.0)

        # we can't be sure of the exact values, but we generally expect
        # a higher standard deviation with T=5
        std_low = velocities_lowT.std().item()
        std_high = velocities_highT.std().item()

        # check that highT typically yields a bigger spread
        self.assertTrue(
            std_high > std_low, "Temperature scaling didn't increase spread."
        )

    def test_sample_gmm_single_component(self):
        """
        Edge case: K=1 => single-component mixture
        Should sample exactly from that one Gaussian => effectively the same for each node's single mixture.
        """
        N = 5
        d = 3
        K = 1
        per_comp = d + (d * (d + 1)) // 2 + 1
        out_dim = K * per_comp
        network_output = torch.randn(N, out_dim)
        # This means no "mixture" choice is needed, comp_ids always 0

        velocities = sample_gmm(network_output, d=d, K=K, temperature=1.0)
        self.assertEqual(velocities.shape, (N, d))


if __name__ == "__main__":
    unittest.main()
