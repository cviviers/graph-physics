from functools import partial
from typing import Callable, List, Optional, Union

import torch
import torch_geometric.transforms as T
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from graphphysics.utils.nodetype import NodeType


def add_edge_features() -> List[Callable[[Data], Data]]:
    """
    Returns a list of PyTorch Geometric transforms to add edge features to a graph.

    Returns:
        List[Callable[[Data], Data]]: List of transforms to add edge features.
    """
    return [T.Cartesian(norm=False), T.Distance(norm=False)]


def _3d_face_to_edge(graph: Data) -> Data:
    """
    Converts 3D quadrilateral faces to triangular faces.

    Parameters:
        graph (Data): The input graph data.

    Returns:
        Data: The graph with updated faces.
    """
    face = graph.face
    graph.face = torch.cat(
        [
            face[0:3],
            face[1:4],
            torch.stack([face[2], face[3], face[0]], dim=0),
            torch.stack([face[3], face[0], face[1]], dim=0),
        ],
        dim=1,
    )
    return graph


def add_obstacles_next_pos(
    graph: Data,
    world_pos_index_start: int,
    world_pos_index_end: int,
    node_type_index: int,
) -> Data:
    """
    Adds obstacle displacement to node features in the graph.

    Parameters:
        graph (Data): The input graph data.
        world_pos_index_start (int): The starting index of world position in node features.
        world_pos_index_end (int): The ending index of world position in node features.
        node_type_index (int): The index of the node type feature.

    Returns:
        Data: The graph with updated node features.
    """
    # Extract world positions and other features
    world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
    other_features = graph.x[:, world_pos_index_end:]

    # Extract target world positions from graph.y
    target_world_pos = graph.y[:, world_pos_index_start:world_pos_index_end]

    # Compute obstacle displacement
    obstacle_displacement = target_world_pos - world_pos

    # Get node types
    # -3 because the index we gave will be the proper index after we added the
    # dimensionals obstacle next pos
    node_type = graph.x[:, node_type_index - 3]

    # Create mask for nodes that are not obstacles
    mask = node_type != NodeType.OBSTACLE
    obstacle_displacement[mask] = 0

    # Update node features
    graph.x = torch.cat([world_pos, obstacle_displacement, other_features], dim=1)
    return graph


def add_world_edges(
    graph: Data,
    world_pos_index_start: int,
    world_pos_index_end: int,
    node_type_index: int,
    radius: float = 0.03,
) -> Data:
    """
    Adds world edges to the graph based on proximity in world position.

    Parameters:
        graph (Data): The input graph data.
        world_pos_index_start (int): The starting index of world position in node features.
        world_pos_index_end (int): The ending index of world position in node features.
        node_type_index (int): The index of the node type feature.
        radius (float): The radius within which to connect nodes.

    Returns:
        Data: The graph with added world edges.
    """

    # Extract world positions
    def _close_pairs_ckdtree(X, max_d):
        tree = cKDTree(X.cpu().numpy())
        pairs = tree.query_pairs(max_d, output_type="ndarray")
        return torch.Tensor(pairs.T).long()

    world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
    added_edges = _close_pairs_ckdtree(world_pos, radius)

    type = graph.x[:, node_type_index]

    m1 = torch.gather(type, -1, added_edges[0]) == NodeType.NORMAL
    m2 = torch.gather(type, -1, added_edges[1]) == NodeType.NORMAL
    mask = torch.logical_and(m1, m2)

    added_edges = added_edges[:, mask]

    edge_index = torch.cat([added_edges, graph.edge_index], dim=1)
    edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)

    graph.edge_index = edge_index
    return graph


def add_world_pos_features(
    graph: Data,
    world_pos_index_start: int,
    world_pos_index_end: int,
) -> Data:
    """
    Adds world position features to the graph's edge attributes.

    Parameters:
        graph (Data): The input graph data.
        world_pos_index_start (int): The starting index of world position in node features.
        world_pos_index_end (int): The ending index of world position in node features.

    Returns:
        Data: The graph with updated edge attributes.
    """
    world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
    senders, receivers = graph.edge_index

    relative_world_pos = world_pos[senders] - world_pos[receivers]
    relative_world_pos_norm = torch.norm(relative_world_pos, p=2, dim=-1, keepdim=True)

    graph.edge_attr = torch.cat(
        [
            graph.edge_attr,
            relative_world_pos.type_as(graph.edge_attr),
            relative_world_pos_norm.type_as(graph.edge_attr),
        ],
        dim=-1,
    )

    return graph


def add_noise(
    graph: Data,
    noise_index_start: Union[int, List[int]],
    noise_index_end: Union[int, List[int]],
    noise_scale: Union[float, List[float]],
    node_type_index: int,
) -> Data:
    """
    Adds Gaussian noise to the specified features of the graph's nodes.

    Parameters:
        graph (Data): The graph to modify.
        noise_index_start (Union[int, List[int]]): The starting index or indices for noise addition.
        noise_index_end (Union[int, List[int]]): The ending index or indices for noise addition.
        noise_scale (Union[float, List[float]]): The standard deviation(s) of the Gaussian noise.
        node_type_index (int): The index of the node type feature.

    Returns:
        Data: The modified graph with noise added to node features.
    """
    # Ensure noise indices are lists
    if isinstance(noise_index_start, int):
        noise_index_start = [noise_index_start]
    if isinstance(noise_index_end, int):
        noise_index_end = [noise_index_end]

    # Ensure noise scales are lists
    if isinstance(noise_scale, float):
        noise_scale = [noise_scale] * len(noise_index_start)

    if len(noise_index_start) != len(noise_index_end):
        raise ValueError(
            "noise_index_start and noise_index_end must have the same length."
        )
    if len(noise_scale) != len(noise_index_start):
        raise ValueError(
            "noise_scale must have the same length as noise_index_start and noise_index_end."
        )

    node_type = graph.x[:, node_type_index]

    # Mask to zero noise for nodes that are not NORMAL
    mask = node_type != NodeType.NORMAL

    for start, end, scale in zip(noise_index_start, noise_index_end, noise_scale):
        feature = graph.x[:, start:end]

        # Generate noise
        noise = torch.randn_like(feature) * scale

        # Zero out noise for nodes not of type NORMAL
        noise[mask] = 0

        # Add noise to features
        graph.x[:, start:end] = feature + noise

    return graph


def build_preprocessing(
    noise_parameters: Optional[dict] = None,
    world_pos_parameters: Optional[dict] = None,
    add_edges_features: bool = True,
    extra_node_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
    extra_edge_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
) -> T.Compose:
    """
    Builds a preprocessing transform pipeline for the graph data.

    Parameters:
        noise_parameters (dict, optional): Parameters for adding noise.
        world_pos_parameters (dict, optional): Parameters for adding world position features.
        add_edges_features (bool): Whether to add edge features.
        extra_node_features (Callable or List[Callable], optional): Extra node feature functions to apply first.
        extra_edge_features (Callable or List[Callable], optional): Extra edge feature functions to apply last.

    Returns:
        T.Compose: A composition of graph transformations.
    """
    preprocessing: List[Callable[[Data], Data]] = []

    # Add extra node features functions at the beginning
    if extra_node_features is not None:
        if not isinstance(extra_node_features, list):
            extra_node_features = [extra_node_features]
        preprocessing.extend(extra_node_features)

    if world_pos_parameters is not None:
        preprocessing.extend(
            [
                partial(
                    add_obstacles_next_pos,
                    world_pos_index_start=world_pos_parameters["world_pos_index_start"],
                    world_pos_index_end=world_pos_parameters["world_pos_index_end"],
                    node_type_index=world_pos_parameters["node_type_index"],
                ),
                _3d_face_to_edge,
                T.FaceToEdge(),
                partial(
                    add_world_edges,
                    world_pos_index_start=world_pos_parameters["world_pos_index_start"],
                    world_pos_index_end=world_pos_parameters["world_pos_index_end"],
                    node_type_index=world_pos_parameters["node_type_index"],
                    radius=world_pos_parameters.get("radius", 0.03),
                ),
            ]
        )
        preprocessing.extend(add_edge_features())
        preprocessing.append(
            partial(
                add_world_pos_features,
                world_pos_index_start=world_pos_parameters["world_pos_index_start"],
                world_pos_index_end=world_pos_parameters["world_pos_index_end"],
            )
        )
    else:
        preprocessing.append(T.FaceToEdge())
        if add_edges_features:
            preprocessing.extend(add_edge_features())

    if noise_parameters is not None:
        add_noise_transform = partial(
            add_noise,
            noise_index_start=noise_parameters["noise_index_start"],
            noise_index_end=noise_parameters["noise_index_end"],
            noise_scale=noise_parameters["noise_scale"],
            node_type_index=noise_parameters["node_type_index"],
        )
        # Insert after the first transform
        preprocessing.insert(1, add_noise_transform)

    # Append extra edge features functions at the end
    if extra_edge_features is not None:
        if not isinstance(extra_edge_features, list):
            extra_edge_features = [extra_edge_features]
        preprocessing.extend(extra_edge_features)

    return T.Compose(preprocessing)
