from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch_geometric.data import Data

from graphphysics.dataset.h5_dataset import H5Dataset
from graphphysics.dataset.preprocessing import build_preprocessing
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from graphphysics.models.processors import EncodeProcessDecode, EncodeTransformDecode
from graphphysics.models.simulator import Simulator
from graphphysics.utils.nodetype import NodeType


def get_preprocessing(
    param: Dict[str, Any],
    device: torch.device,
    use_edge_feature: bool = True,
    extra_node_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
    extra_edge_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
):
    """
    Constructs the preprocessing function based on provided parameters.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        device (torch.device): The device to perform computations on.
        use_edge_feature (bool, optional): Whether to add edge features. Defaults to True.
        extra_node_features (Optional[Union[Callable[[Data], Data], List[Callable[[Data], Data]]]], optional):
            Additional functions to compute extra node features. Defaults to None.
        extra_edge_features (Optional[Union[Callable[[Data], Data], List[Callable[[Data], Data]]]], optional):
            Additional functions to compute extra edge features. Defaults to None.

    Returns:
        Callable[[Data], Data]: A function that preprocesses a Data object.
    """
    preprocessing_params = param.get("transformations", {}).get("preprocessing", {})
    noise_scale = preprocessing_params.get("noise", 0)
    noise_parameters = None
    if noise_scale != 0:
        noise_parameters = {
            "noise_index_start": preprocessing_params.get("noise_index_start"),
            "noise_index_end": preprocessing_params.get("noise_index_end"),
            "noise_scale": noise_scale,
            "node_type_index": param["index"]["node_type_index"],
        }

    world_pos_params = param.get("transformations", {}).get("world_pos_parameters", {})
    world_pos_parameters = None
    if world_pos_params.get("use", False):
        world_pos_parameters = {
            "world_pos_index_start": world_pos_params.get("world_pos_index_start"),
            "world_pos_index_end": world_pos_params.get("world_pos_index_end"),
            "node_type_index": param["index"]["node_type_index"],
        }

    return build_preprocessing(
        noise_parameters=noise_parameters,
        world_pos_parameters=world_pos_parameters,
        add_edges_features=use_edge_feature,
        extra_node_features=extra_node_features,
        extra_edge_features=extra_edge_features,
    )


def get_model(param: Dict[str, Any], only_processor: bool = False):
    """
    Constructs the model based on provided parameters.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        only_processor (bool, optional): Whether to use only the processor part of the model. Defaults to False.

    Returns:
        nn.Module: The constructed model.

    Raises:
        ValueError: If the model type specified in param is not supported.
    """
    model_type = param.get("model", {}).get("type", "")
    node_input_size = param["model"]["node_input_size"] + NodeType.SIZE

    num_mixture_components = param["model"].get("num_mixture_components", 0)
    temperature = param["model"].get("temperature")

    if model_type == "epd":
        return EncodeProcessDecode(
            message_passing_num=param["model"]["message_passing_num"],
            node_input_size=node_input_size,
            edge_input_size=param["model"]["edge_input_size"],
            output_size=param["model"]["output_size"],
            hidden_size=param["model"]["hidden_size"],
            only_processor=only_processor,
            num_mixture_components=num_mixture_components,
            temperature=temperature,
        )
    elif model_type == "transformer":
        return EncodeTransformDecode(
            message_passing_num=param["model"]["message_passing_num"],
            node_input_size=node_input_size,
            output_size=param["model"]["output_size"],
            hidden_size=param["model"]["hidden_size"],
            num_heads=param["model"]["num_heads"],
            only_processor=only_processor,
            num_mixture_components=num_mixture_components,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")


def get_simulator(param: Dict[str, Any], model, device: torch.device) -> Simulator:
    """
    Constructs the Simulator based on provided parameters.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        model: The model to be used within the simulator.
        device (torch.device): The device to perform computations on.

    Returns:
        Simulator: The constructed Simulator object.
    """
    node_input_size = param["model"]["node_input_size"] + NodeType.SIZE

    return Simulator(
        node_input_size=node_input_size,
        edge_input_size=param["model"]["edge_input_size"],
        output_size=param["model"]["output_size"],
        feature_index_start=param["index"]["feature_index_start"],
        feature_index_end=param["index"]["feature_index_end"],
        output_index_start=param["index"]["output_index_start"],
        output_index_end=param["index"]["output_index_end"],
        node_type_index=param["index"]["node_type_index"],
        model=model,
        device=device,
    )


def get_dataset(
    param: Dict[str, Any],
    preprocessing: Callable[[Data], Data],
    masking_ratio: Optional[float] = None,
    use_edge_feature: bool = True,
    use_previous_data: bool = False,
    switch_to_val: bool = False,
):
    """
    Constructs the dataset based on provided parameters.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        preprocessing (Callable[[Data], Data]): The preprocessing function to apply to the data.
        masking_ratio (Optional[float], optional): The ratio of data to mask. Defaults to None.
        use_edge_feature (bool, optional): Whether to add edge features. Defaults to True.
        use_previous_data (bool, optional): Whether to use previous data in the dataset. Defaults to False.

    Returns:
        Dataset: The constructed dataset.

    Raises:
        ValueError: If the dataset extension specified in param is not supported.
    """
    dataset_params = param.get("dataset", {})
    khop = dataset_params.get("khop", 1)
    new_edges_ratio = dataset_params.get("new_edges_ratio", 0)
    extension = dataset_params.get("extension", "")

    world_pos_parameters = None
    if khop > 1:
        transformations = param.get("transformations", {})
        if "world_pos_parameters" in transformations:
            wpp = transformations["world_pos_parameters"]
            if wpp.get("use", False):
                world_pos_parameters = wpp

    if extension == "h5":
        return H5Dataset(
            h5_path=dataset_params["h5_path"],
            meta_path=dataset_params["meta_path"],
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=use_edge_feature,
            use_previous_data=use_previous_data,
            switch_to_val=switch_to_val,
            world_pos_parameters=world_pos_parameters,
        )
    elif extension == "xdmf":
        return XDMFDataset(
            xdmf_folder=dataset_params["xdmf_folder"],
            meta_path=dataset_params["meta_path"],
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=use_edge_feature,
            use_previous_data=use_previous_data,
            switch_to_val=switch_to_val,
        )
    else:
        raise ValueError(f"Dataset extension '{extension}' not supported.")


def get_num_workers(param: Dict[str, Any], default_num_workers: int) -> int:
    """
    Determines the number of workers to use for DataLoader based on dataset extension.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        default_num_workers (int): The default number of workers specified.

    Returns:
        int: The adjusted number of workers.
    """
    dataset_params = param.get("dataset", {})
    extension = dataset_params.get("extension", "")
    if extension == "h5":
        return 0
    elif extension == "xdmf":
        return default_num_workers
    else:
        raise ValueError(f"Dataset extension '{extension}' not supported.")
