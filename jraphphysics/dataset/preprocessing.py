import jax
import jax.numpy as jnp
import jraph
from typing import Union, Tuple, Optional


def add_noise(
    graph: jraph.GraphsTuple,
    noise_index_start: Union[int, Tuple[int, ...]],
    noise_index_end: Union[int, Tuple[int, ...]],
    noise_scale: Union[float, Tuple[float, ...]],
    node_type_index: int,
    key: jax.random.PRNGKey,
    node_type_normal_value: int = 0,
) -> Tuple[jraph.GraphsTuple, jax.random.PRNGKey]:
    """
    Adds Gaussian noise to the specified features of the graph's nodes.

    Parameters:
        graph (jraph.GraphsTuple): The graph to modify.
        noise_index_start (Union[int, Tuple[int, ...]]): The starting index or indices for noise addition.
        noise_index_end (Union[int, Tuple[int, ...]]): The ending index or indices for noise addition.
        noise_scale (Union[float, Tuple[float, ...]]): The standard deviation(s) of the Gaussian noise.
        node_type_index (int): The index of the node type feature.
        key (jax.random.PRNGKey): The random key for noise generation.
        node_type_normal_value (int, optional): The value representing NodeType.NORMAL. Defaults to 0.

    Returns:
        Tuple[jraph.GraphsTuple, jax.random.PRNGKey]: The modified graph with noise added to node features and the new random key.
    """
    # Ensure noise indices are tuples
    if isinstance(noise_index_start, int):
        noise_index_start = (noise_index_start,)
    if isinstance(noise_index_end, int):
        noise_index_end = (noise_index_end,)

    # Ensure noise scales are tuples
    if isinstance(noise_scale, float):
        noise_scale = (noise_scale,) * len(noise_index_start)

    if len(noise_index_start) != len(noise_index_end):
        raise ValueError(
            "noise_index_start and noise_index_end must have the same length."
        )
    if len(noise_scale) != len(noise_index_start):
        raise ValueError(
            "noise_scale must have the same length as noise_index_start and noise_index_end."
        )

    nodes = graph.nodes
    features = nodes["features"]  # Assuming node features are stored under 'features'

    node_type = features[:, node_type_index]

    # Mask to zero noise for nodes that are not NORMAL
    mask = (
        node_type == node_type_normal_value
    )  # Boolean array where True indicates NORMAL nodes
    mask = mask.astype(features.dtype)[:, None]  # Expand dims to match features shape

    # Initialize new_features with the original features
    new_features = features

    # Loop over the noise ranges
    for start, end, scale in zip(noise_index_start, noise_index_end, noise_scale):
        slice_size = end - start
        noise_shape = (features.shape[0], slice_size)
        key, subkey = jax.random.split(key)
        # Generate noise with static shape
        noise = jax.random.normal(subkey, shape=noise_shape) * scale
        # Apply mask
        noise = noise * mask  # Apply mask to zero out noise for nodes not NORMAL
        # Update new_features
        new_features = new_features.at[:, start:end].add(noise)

    # Create new nodes dictionary
    new_nodes = dict(nodes)
    new_nodes["features"] = new_features

    # Return a new graph with updated nodes
    new_graph = graph._replace(nodes=new_nodes)

    return new_graph, key
