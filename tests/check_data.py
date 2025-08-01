# get the shapes of the h5 datasets 
import h5py
import os

data_path = r"C:\Users\20195435\Documents\TUe\Tasti\graph-physics\dataset\h5_dataset\deforming_plate\train.h5"

def get_h5_shapes(file_path: str):
    """
    Get the shapes of datasets in an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary with dataset names as keys and their shapes as values.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    shapes = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                shapes[key] = f[key].shape
            elif isinstance(f[key], h5py.Group):
                for sub_key in f[key].keys():
                    sub_item = f[key][sub_key]
                    if isinstance(sub_item, h5py.Dataset):
                        shapes[f"{key}/{sub_key}"] = sub_item.shape
    return shapes

if __name__ == "__main__":
    shapes = get_h5_shapes(data_path)
    print("Shapes of datasets in the HDF5 file:")
    for name, shape in shapes.items():
        print(f"{name}: {shape}")