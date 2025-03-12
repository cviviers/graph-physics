import numpy as np
import os
import matplotlib.pyplot as plt
import random
import trimesh
from sklearn.neighbors import NearestNeighbors
import vtk


def compute_best_fit_transform(source_points, target_points):
    """
    Compute the rigid transformation that best aligns the source points to the target points.

    Parameters:
    - source_points: Numpy array of shape (N, D) representing the source points.
    - target_points: Numpy array of shape (N, D) representing the target points.

    Returns:
    - rotation_matrix: Rotation matrix of shape (D, D).
    - translation_vector: Translation vector of shape (D,).
    """
    assert (
        source_points.shape[1] == target_points.shape[1]
    ), "Source and target must have the same dimensionality."

    # Compute centroids of the source and target points
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Center the points by subtracting the centroid
    centered_source = source_points - source_centroid
    centered_target = target_points - target_centroid

    # Compute the covariance matrix
    covariance_matrix = np.dot(centered_source.T, centered_target)

    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Compute the rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)

    # Handle reflection case
    if np.linalg.det(rotation_matrix) < 0:
        Vt[2, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)

    # Compute the translation vector
    translation_vector = target_centroid - np.dot(rotation_matrix, source_centroid)

    return rotation_matrix, translation_vector


def iterative_closest_point(
    source_cloud, target_cloud, max_iterations=20, tolerance=0.001
):
    """
    Align the source point cloud to the target point cloud using the Iterative Closest Point (ICP) algorithm.

    Parameters:
    - source_cloud: Numpy array of shape (M, D) representing the source point cloud.
    - target_cloud: Numpy array of shape (N, D) representing the target point cloud.
    - max_iterations: Maximum number of iterations for the ICP algorithm.
    - tolerance: Convergence criterion based on change in alignment error.

    Returns:
    - aligned_source: Transformed source point cloud aligned to the target.
    - cumulative_rotation: Cumulative rotation matrix.
    - cumulative_translation: Cumulative translation vector.
    """
    # Initialize cumulative transformations
    cumulative_rotation = np.eye(source_cloud.shape[1])
    cumulative_translation = np.zeros(source_cloud.shape[1])

    previous_error = 0

    for iteration in range(max_iterations):
        # Find the closest points in the target cloud for each point in the source cloud
        nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(target_cloud)
        distances, indices = nearest_neighbors.kneighbors(source_cloud)
        closest_points = target_cloud[indices.flatten()]

        # Compute the best-fit transform
        rotation_matrix, translation_vector = compute_best_fit_transform(
            source_cloud, closest_points
        )

        # Update the cumulative transformations
        cumulative_rotation = rotation_matrix @ cumulative_rotation
        cumulative_translation = (
            translation_vector + rotation_matrix @ cumulative_translation
        )

        # Apply the transformation to the source cloud
        source_cloud = np.dot(source_cloud, rotation_matrix.T) + translation_vector

        # Compute the mean error
        mean_error = np.mean(distances.flatten())

        # Check for convergence
        if abs(previous_error - mean_error) < tolerance:
            break
        previous_error = mean_error

    return source_cloud, cumulative_rotation, cumulative_translation


def load_files():
    coarse_dataset_folder = "surface_dataset"
    latents_folder = (
        "dataset_output/latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
    )

    # Check if the coarse_dataset folder exists
    if not os.path.exists(coarse_dataset_folder):
        raise FileNotFoundError(f"The folder {coarse_dataset_folder} does not exist.")

    # Get list of obj files in the coarse_dataset folder
    obj_files = [f for f in os.listdir(coarse_dataset_folder) if f.endswith(".obj")]
    if len(obj_files) == 0:
        raise FileNotFoundError("No obj files found in the coarse_dataset folder.")

    return obj_files, coarse_dataset_folder, latents_folder


def save_vtk(points, filename):
    """
    Save points to a VTK file.

    Parameters:
    - points: Numpy array of shape (N, 3) representing the points.
    - filename: Path to the output VTK file.
    """
    polydata = vtk.vtkPolyData()
    points_vtk = vtk.vtkPoints()
    for point in points:
        points_vtk.InsertNextPoint(point)
    polydata.SetPoints(points_vtk)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def plot_data(obj_path, npz_path, output_folder):
    # Load obj file
    mesh = trimesh.load(obj_path)
    obj_points = mesh.vertices

    # Load npz file
    npz_data = np.load(npz_path)
    npz_points = npz_data["coords"]

    # Permute the second and third columns of npz_points
    npz_points[:, [1, 2]] = npz_points[:, [2, 1]]

    # Rescale npz_points to match the scale of obj_points
    npz_points = scale_pos(obj_points, npz_points)

    # Apply symmetry with respect to the x-y plane
    npz_points[:, 2] = -npz_points[:, 2]

    # Compute centroids of obj_points and npz_points
    obj_centroid = np.mean(obj_points, axis=0)
    npz_centroid = np.mean(npz_points, axis=0)

    # Compute the translation vector to align the centroids
    translation_vector = obj_centroid - npz_centroid

    # Apply the translation to npz_points
    npz_points[:, 2] += translation_vector[2]

    # Print the translation vector
    print("Translation vector to align centroids:", translation_vector)

    # Apply ICP to align npz_points to obj_points
    # npz_points_aligned, _, _  = iterative_closest_point(npz_points, obj_points)

    # Save the original and aligned points to VTK files
    obj_filename = os.path.join(
        output_folder,
        os.path.splitext(os.path.basename(obj_path))[0] + "_obj_points.vtk",
    )
    npz_filename = os.path.join(
        output_folder,
        os.path.splitext(os.path.basename(npz_path))[0] + "_npz_points_aligned.vtk",
    )
    save_vtk(obj_points, obj_filename)
    save_vtk(npz_points, npz_filename)


def scale_pos(obj_points: np.ndarray, coords: np.ndarray):
    obj_min, obj_max = obj_points.min(axis=0), obj_points.max(axis=0)
    coords_min, coords_max = coords.min(axis=0), coords.max(axis=0)

    scale = (obj_max - obj_min) / (coords_max - coords_min)
    shift = obj_min - coords_min * scale
    coords_scaled = coords * scale + shift

    return coords_scaled


# Main execution
if __name__ == "__main__":
    obj_files, coarse_dataset_folder, latents_folder = load_files()
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for obj_file in obj_files:
        obj_path = os.path.join(coarse_dataset_folder, obj_file)
        npz_file_name = os.path.splitext(obj_file)[0] + ".npz"
        npz_path = os.path.join(latents_folder, npz_file_name)

        if not os.path.exists(npz_path):
            print(f"Corresponding npz file not found for {obj_file}.")
            continue

        plot_data(obj_path, npz_path, output_folder)
