import os
import json
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import torch
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from graphphysics.utils.torch_graph import meshdata_to_graph

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_vtk(points, filename):
    import vtk
    polydata = vtk.vtkPolyData()
    points_vtk = vtk.vtkPoints()
    for point in points:
        points_vtk.InsertNextPoint(point)
    polydata.SetPoints(points_vtk)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

def plot_data(xdmf_dataset, index, output_folder):
    graph, _ = xdmf_dataset[index]
    feats, coords = xdmf_dataset.get_endcoding(index=index)

    # Rescale coords to match the scale of graph positions
    coords = xdmf_dataset.scale_pos(graph, coords)

    # Apply symmetry with respect to the x-y plane
    coords[:, 2] = -coords[:, 2]

    # Compute centroids of graph positions and coords
    graph_centroid = np.mean(graph.pos.numpy(), axis=0)
    coords_centroid = np.mean(coords, axis=0)

    # Compute the translation vector to align the centroids
    translation_vector = graph_centroid - coords_centroid

    # Apply the translation to coords
    coords[:, 2] += translation_vector[2]

    # Print the translation vector
    print("Translation vector to align centroids:", translation_vector)

    # Save the original and aligned points to VTK files
    graph_filename = os.path.join(output_folder, f"graph_points_{index}.vtk")
    coords_filename = os.path.join(output_folder, f"coords_points_{index}.vtk")
    save_vtk(graph.pos.numpy(), graph_filename)
    save_vtk(coords, coords_filename)

# Main execution
if __name__ == "__main__":
    config_path = "training_config/coarse-aneurysm.json"
    config = load_config(config_path)
    
    xdmf_folder = "coarse_dataset"
    meta_path = config["dataset"]["meta_path"]
    masking_ratio = config["dataset"]["masking_ratio"]
    khop = config["dataset"]["khop"]
    add_edge_features = False
    node_type_index = config["index"]["node_type_index"]

    
    xdmf_dataset = XDMFDataset(
        xdmf_folder=xdmf_folder,
        meta_path=meta_path,
        masking_ratio=masking_ratio,
        khop=khop,
        add_edge_features=add_edge_features,
        node_type_index=node_type_index,
    ) 

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for index in range(len(xdmf_dataset)):
        plot_data(xdmf_dataset, index, output_folder)
