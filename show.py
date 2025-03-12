import os
import json
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import torch
import random
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from graphphysics.utils.torch_graph import meshdata_to_graph


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def plot_data(xdmf_dataset, index):
    graph = xdmf_dataset[index * xdmf_dataset.trajectory_length]

    file = xdmf_dataset.file_paths[index]
    file_name = file.replace(".xdmf", "")
    print(f"File name without extension: {file_name}")

    xdmf_dataset.plot_rescaled(graph, index, file_name)


# Main execution
if __name__ == "__main__":
    config_path = "training_config/coarse-aneurysm.json"
    config = load_config(config_path)

    xdmf_folder = config["dataset"]["xdmf_folder"]
    meta_path = config["dataset"]["meta_path"]
    khop = config["dataset"]["khop"]
    add_edge_features = False
    node_type_index = config["index"]["node_type_index"]

    xdmf_dataset = XDMFDataset(
        xdmf_folder=xdmf_folder,
        meta_path=meta_path,
        khop=khop,
        add_edge_features=add_edge_features,
        node_type_index=node_type_index,
    )

    # Process one random object from the dataset
    random_index = random.randint(0, xdmf_dataset._size_dataset - 1)
    plot_data(xdmf_dataset, random_index)
