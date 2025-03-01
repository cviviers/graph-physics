import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def read_random_npz_in_latents():
    latents_dir = "latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"

    # List all files in the latents subdirectory
    files = os.listdir(latents_dir)

    # Filter out the .npz files
    npz_files = [f for f in files if f.endswith('.npz')]

    if not npz_files:
        raise FileNotFoundError("No .npz files found in the latents subdirectory.")

    # Get a random .npz file
    random_npz_file = random.choice(npz_files)
    random_npz_path = os.path.join(latents_dir, random_npz_file)

    # Load the .npz file
    data = np.load(random_npz_path)

    # Print the names of the arrays in the .npz file
    print("Arrays in the .npz file:", data.files)

    # Read the matrices (assuming they are stored as arrays in the .npz file)
    matrices = {name: data[name] for name in data.files}

    return matrices

if __name__ == "__main__":
    matrices = read_random_npz_in_latents()

    # Retrieve the matrices and display the data of the second one in 3D
    if len(matrices) < 2:
        raise ValueError("Less than two matrices found in the .npz file.")

    second_matrix_name = list(matrices.keys())[1]
    second_matrix = matrices[second_matrix_name]

    print(f"Data of the second matrix '{second_matrix_name}' in 3D:")
    print(second_matrix)

    # Plot the data of the second matrix in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assuming the second matrix has shape (n, 3) where n is the number of points
    x = second_matrix[:, 0]
    y = second_matrix[:, 1]  # Diviser par 20 pour l'axe Y
    z = second_matrix[:, 2]

    ax.scatter(x, y, z)

    # Set labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Calculate limits for equal aspect ratio
    max_range = np.max([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]) / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
