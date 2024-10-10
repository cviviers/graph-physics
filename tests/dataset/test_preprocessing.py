import unittest
from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

from graphphysics.utils.nodetype import NodeType
from graphphysics.dataset.preprocessing import (
    add_edge_features,
    add_obstacles_next_pos,
    add_world_edges,
    add_world_pos_features,
    add_noise,
    build_preprocessing,
)


class TestGraphPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a simple graph for testing
        self.num_nodes = 4
        self.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
        self.x = torch.tensor(
            [
                [0.0, 1.0, 2.0, NodeType.NORMAL],
                [1.0, 2.0, 3.0, NodeType.OBSTACLE],
                [2.0, 3.0, 4.0, NodeType.NORMAL],
                [3.0, 4.0, 5.0, NodeType.NORMAL],
            ],
            dtype=torch.float32,
        )
        self.pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.y = torch.tensor(
            [
                [0.1, 1.1, 2.1],
                [1.1, 2.1, 3.1],
                [2.1, 3.1, 4.1],
                [3.1, 4.1, 5.1],
            ],
            dtype=torch.float32,
        )
        self.graph = Data(x=self.x, edge_index=self.edge_index, y=self.y, pos=self.pos)

    def test_add_edge_features(self):
        transform = Compose(add_edge_features())
        graph = transform(self.graph.clone())
        self.assertIsNotNone(graph.edge_attr)
        self.assertEqual(graph.edge_attr.shape[1], 4)

    def test_add_obstacles_next_pos(self):
        graph = self.graph.clone()
        graph = add_obstacles_next_pos(
            graph,
            world_pos_index_start=0,
            world_pos_index_end=3,
            node_type_index=3,
        )
        self.assertEqual(graph.x.shape[1], self.x.shape[1] + 3)

    def test_add_world_edges(self):
        graph = self.graph.clone()
        graph = add_world_edges(
            graph,
            world_pos_index_start=0,
            world_pos_index_end=3,
            node_type_index=3,
            radius=5.0,  # Large radius to connect all nodes
        )
        # Check that edge_index has been updated
        self.assertGreater(graph.edge_index.shape[1], self.edge_index.shape[1])

    def test_add_world_pos_features(self):
        graph = self.graph.clone()
        # Assume edge_attr is initialized
        graph.edge_attr = torch.zeros((graph.edge_index.shape[1], 1))
        graph = add_world_pos_features(
            graph,
            world_pos_index_start=0,
            world_pos_index_end=3,
        )
        # Check that edge_attr has increased dimensions
        self.assertEqual(
            graph.edge_attr.shape[1], 5
        )  # Original 1 + 3 for relative position + 1 norm

    def test_add_noise(self):
        graph = self.graph.clone()
        graph = add_noise(
            graph,
            noise_index_start=0,
            noise_index_end=3,
            noise_scale=0.1,
            node_type_index=3,
        )
        # Check that node features have changed for NORMAL nodes
        normal_indices = (self.x[:, 3] == NodeType.NORMAL).nonzero(as_tuple=True)[0]
        self.assertFalse(
            torch.allclose(graph.x[normal_indices, 0:3], self.x[normal_indices, 0:3])
        )

        # Check that node features have not changed for non-NORMAL nodes
        non_normal_indices = (self.x[:, 3] != NodeType.NORMAL).nonzero(as_tuple=True)[0]
        self.assertTrue(
            torch.allclose(
                graph.x[non_normal_indices, 0:3], self.x[non_normal_indices, 0:3]
            )
        )

    def test_build_preprocessing(self):
        noise_params = {
            "noise_index_start": 0,
            "noise_index_end": 3,
            "noise_scale": 0.1,
            "node_type_index": 3,
        }
        world_pos_params = {
            "world_pos_index_start": 0,
            "world_pos_index_end": 3,
            "node_type_index": 3,
            "radius": 5.0,
        }
        transform = build_preprocessing(noise_params, world_pos_params)
        graph = transform(
            Data(
                x=self.x,
                face=torch.tensor([[0], [1], [2], [3]], dtype=torch.int),
                y=self.y,
                pos=self.pos,
            )
        )
        self.assertIsNotNone(graph.edge_index)
        self.assertIsNotNone(graph.edge_attr)
        self.assertEqual(graph.x.shape[1], self.x.shape[1] + 3)

    def test_add_edge_features_without_world_pos(self):
        # Test add_edge_features when world_pos_parameters is None
        transform = build_preprocessing(add_edges_features=True)
        graph = transform(
            Data(
                x=self.x,
                face=torch.tensor([[0], [1], [2]], dtype=torch.int),
                y=self.y,
                pos=self.pos,
            ).clone()
        )
        self.assertIsNotNone(graph.edge_attr)
        self.assertEqual(
            graph.edge_attr.shape[1], 4
        )  # Assuming 3 for Cartesian and 1 for Distance

    def test_build_preprocessing_without_noise(self):
        world_pos_params = {
            "world_pos_index_start": 0,
            "world_pos_index_end": 3,
            "node_type_index": 3,
            "radius": 5.0,
        }
        transform = build_preprocessing(world_pos_parameters=world_pos_params)
        graph = transform(
            Data(
                x=self.x,
                face=torch.tensor([[0], [1], [2], [3]], dtype=torch.int),
                y=self.y,
                pos=self.pos,
            )
        )
        self.assertIsNotNone(graph.edge_index)
        self.assertIsNotNone(graph.edge_attr)

    def test_build_preprocessing_without_world_pos(self):
        noise_params = {
            "noise_index_start": 0,
            "noise_index_end": 3,
            "noise_scale": 0.1,
            "node_type_index": 3,
        }
        transform = build_preprocessing(noise_parameters=noise_params)
        graph = transform(
            Data(
                x=self.x,
                face=torch.tensor([[0], [1], [2]], dtype=torch.int),
                y=self.y,
                pos=self.pos,
            )
        )
        self.assertIsNotNone(graph.edge_index)
        self.assertIsNotNone(graph.edge_attr)

    def test_add_noise_with_multiple_indices(self):
        graph = Data(
            x=self.x,
            face=torch.tensor([[0], [1], [2]], dtype=torch.int),
            y=self.y,
            pos=self.pos,
        )
        noise_params = {
            "noise_index_start": [0, 1],
            "noise_index_end": [1, 3],
            "noise_scale": [10, 20],
            "node_type_index": 3,
        }
        graph = add_noise(
            graph.clone(),
            noise_index_start=noise_params["noise_index_start"],
            noise_index_end=noise_params["noise_index_end"],
            noise_scale=noise_params["noise_scale"],
            node_type_index=noise_params["node_type_index"],
        )
        # Check that node features have changed for NORMAL nodes
        normal_indices = (self.x[:, 3] == NodeType.NORMAL).nonzero(as_tuple=True)[0]
        print(normal_indices)
        self.assertFalse(
            torch.allclose(graph.x[normal_indices, 0:3], self.x[normal_indices, 0:3])
        )

    def test_add_world_edges_with_small_radius(self):
        graph = Data(x=self.x, edge_index=self.edge_index, y=self.y, pos=self.pos)
        graph = add_world_edges(
            graph,
            world_pos_index_start=0,
            world_pos_index_end=3,
            node_type_index=3,
            radius=0.1,  # Small radius to potentially not add new edges
        )
        # Edge index should be same or larger
        self.assertGreaterEqual(graph.edge_index.shape[1], self.edge_index.shape[1])

    def test_add_world_edges_with_large_radius(self):
        graph = Data(x=self.x, edge_index=self.edge_index, y=self.y, pos=self.pos)
        graph = add_world_edges(
            graph.clone(),
            world_pos_index_start=0,
            world_pos_index_end=3,
            node_type_index=3,
            radius=100.0,  # Large radius to connect all nodes
        )
        # Edge index should be larger than original
        self.assertGreater(graph.edge_index.shape[1], self.edge_index.shape[1])

    def test_build_preprocessing_with_extra_features(self):
        # Define extra node and edge feature functions
        def extra_node_feature(graph):
            graph.x = torch.cat(
                (
                    graph.pos,
                    graph.x,
                ),
                dim=1,
            )
            return graph

        noise_params = {
            "noise_index_start": 0,
            "noise_index_end": 3,
            "noise_scale": 0.1,
            "node_type_index": 3,
        }
        world_pos_params = {
            "world_pos_index_start": 0,
            "world_pos_index_end": 3,
            "node_type_index": 3,
            "radius": 5.0,
        }
        transform = build_preprocessing(
            noise_params,
            world_pos_params,
            extra_node_features=extra_node_feature,
        )
        graph = transform(
            Data(
                x=self.x,
                face=torch.tensor([[0], [1], [2], [3]], dtype=torch.int),
                y=self.y,
                pos=self.pos,
            )
        )
        assert graph.x.shape[1] == 10


if __name__ == "__main__":
    unittest.main()
