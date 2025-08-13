import h5py
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any

# view shapes of keys in h5 file

file_path = r"predictions/graph_0.h5"
file_path = r"dataset/h5_dataset/deforming_plate/valid.h5"

with h5py.File(file_path, "r") as f:
    # get the first key
    
    first_key = list(f.keys())[0]
    print(f"First key: {first_key}")
    # get the shape of the elements in the first key
    first = f[first_key]
    print(f"'{first_key}': {first}")
    for key, value in first.items():
        print(f"'{key}': {value.shape}")

    # list all keys and their shapes
    for key in f.keys():
        
        print(f"Key: {key}")
