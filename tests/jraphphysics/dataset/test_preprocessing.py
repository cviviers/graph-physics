import unittest
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from jraphphysics.dataset.preprocessing import add_noise


class TestAddNoise(unittest.TestCase):
    def setUp(self):
        # Set up the PRNG key
        self.key = jax.random.PRNGKey(0)

        # Create a sample graph
        num_nodes = 5
        num_node_features = 4  # e.g., features 0, 1, 2, 3

        # Node features: shape (num_nodes, num_node_features)
        # Let's include a node type at index 3
        node_features = jnp.array(
            [
                [1.0, 2.0, 3.0, 0],  # NORMAL node
                [4.0, 5.0, 6.0, 1],  # Not NORMAL
                [7.0, 8.0, 9.0, 0],  # NORMAL
                [10.0, 11.0, 12.0, 1],  # Not NORMAL
                [13.0, 14.0, 15.0, 0],  # NORMAL
            ],
            dtype=jnp.float32,
        )

        nodes = {"features": node_features}

        # Create dummy edges and other graph components
        senders = jnp.array([0, 1, 2], dtype=jnp.int32)
        receivers = jnp.array([1, 2, 3], dtype=jnp.int32)
        edges = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
        n_node = jnp.array([5], dtype=jnp.int32)
        n_edge = jnp.array([3], dtype=jnp.int32)
        globals = jnp.array([0], dtype=jnp.float32)

        self.graph = jraph.GraphsTuple(
            nodes=nodes,
            senders=senders,
            receivers=receivers,
            edges=edges,
            n_node=n_node,
            n_edge=n_edge,
            globals=globals,
        )

        # Define parameters using tuples
        self.noise_index_start = (0,)
        self.noise_index_end = (2,)  # Add noise to features 0 and 1
        self.noise_scale = (0.1,)
        self.node_type_index = 3
        self.node_type_normal_value = 0  # Nodes with node_type == 0 are NORMAL

    def check_add_noise(self, add_noise_fn):
        # Call add_noise
        new_graph, new_key = add_noise_fn(
            self.graph,
            self.noise_index_start,
            self.noise_index_end,
            self.noise_scale,
            self.node_type_index,
            self.key,
            self.node_type_normal_value,
        )

        # Extract original and new features
        original_features = np.array(self.graph.nodes["features"])
        new_features = np.array(new_graph.nodes["features"])

        # Check that features at indices 0 and 1 have changed for NORMAL nodes
        for i in range(original_features.shape[0]):
            node_type = original_features[i, self.node_type_index]
            if node_type == self.node_type_normal_value:
                # NORMAL node, features should have noise added
                self.assertFalse(
                    np.allclose(original_features[i, 0:2], new_features[i, 0:2])
                )
            else:
                # Not NORMAL, features should be unchanged
                np.testing.assert_allclose(
                    original_features[i, 0:2], new_features[i, 0:2]
                )

            # Features outside the noise range should be unchanged
            np.testing.assert_allclose(
                original_features[i, 2:], new_features[i, 2:], atol=1e-6
            )

        # Check that the new key is different from the original key
        self.assertFalse(np.array_equal(self.key, new_key))

        # Additional check: Ensure that noise is zero where mask is zero
        mask = original_features[:, self.node_type_index] == self.node_type_normal_value
        for i in range(original_features.shape[0]):
            if not mask[i]:
                # Features should be identical
                np.testing.assert_allclose(
                    original_features[i], new_features[i], atol=1e-6
                )
            else:
                # Features may differ due to noise
                pass  # Already checked above

        print("Test passed successfully.")

    def test_add_noise(self):
        # Test the original function
        self.check_add_noise(add_noise)

    def test_add_noise_jit(self):
        # Create a JIT-compiled version of add_noise
        add_noise_jit = jax.jit(
            add_noise,
            static_argnames=[
                "noise_index_start",
                "noise_index_end",
                "noise_scale",
                "node_type_index",
                "node_type_normal_value",
            ],
        )
        # Test the JIT-compiled function
        self.check_add_noise(add_noise_jit)


if __name__ == "__main__":
    unittest.main()
