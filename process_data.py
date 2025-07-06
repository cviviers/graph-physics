import os
import h5py
import numpy as np
import pickle
from typing import Dict, Any, Optional, Union

# evaluate the content of C:\Users\20195435\Documents\TUe\Tasti\graph-physics\dataset\h5_dataset\deforming_plate\train.h5

def evaluate_h5_content(file_path: str) -> Dict[str, Any]:
    """
    Evaluates the content of an HDF5 file and returns a summary of its structure.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        Dict[str, Any]: A dictionary summarizing the structure and content of the HDF5 file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with h5py.File(file_path, 'r') as f:
        summary = {}
        for key in f.keys():
            print(f"Processing key: {key}")
            print(f[key])
            # if type HDF5 group 
            if isinstance(f[key], h5py.Group):

                summary[key] = {
                    'type': 'group',
                    'keys': list(f[key].keys())
                }
                # get the shape of the f[key].keys()
                for sub_key in f[key].keys():
                    sub_item = f[key][sub_key]
                    if isinstance(sub_item, h5py.Dataset):
                        summary[key][sub_key] = {
                            'shape': sub_item.shape,
                            'dtype': str(sub_item.dtype),
                            'size': sub_item.size
                        }
                    elif isinstance(sub_item, h5py.Group):
                        summary[key][sub_key] = {
                            'type': 'group',
                            'keys': list(sub_item.keys())
                        }
            else:
                # For datasets, we summarize their shape, dtype, and size

                summary[key] = {
                    'shape': f[key].shape,
                    'dtype': str(f[key].dtype),
                    'size': f[key].size
                }
    return summary

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\20195435\Documents\TUe\Tasti\graph-physics\dataset\h5_dataset\deforming_plate\train.h5"
    try:
        content_summary = evaluate_h5_content(file_path)
        print("HDF5 File Content Summary:")
        for key, value in content_summary.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"An error occurred: {e}")