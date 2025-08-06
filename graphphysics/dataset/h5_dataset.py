from typing import Callable, Optional, Tuple, Union

import torch
from loguru import logger
from torch_geometric.data import Data

from graphphysics.dataset.dataset import BaseDataset
from graphphysics.utils.hierarchical import (
    get_frame_as_graph,
    get_h5_dataset,
    get_traj_as_meshes,
)


class H5Dataset(BaseDataset):
    def __init__(
        self,
        h5_path: str,
        meta_path: str,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
        world_pos_parameters: Optional[dict] = None,
        use_subset: int = 0,
    ):
        super().__init__(
            meta_path=meta_path,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
            world_pos_parameters=world_pos_parameters,
        )

        if switch_to_val:
            h5_path = h5_path.replace("train", "test")

        self.h5_path = h5_path
        self.meta_path = meta_path
        self.use_subset = use_subset

        self.dt = self.meta["dt"]
        if self.dt == 0:
            self.dt = 1
            logger.warning(
                "The dataset has a timestep set to 0. Fallback to dt=1 to ensure xdmf can be saved."
            )

        # Open the H5 file and load metadata
        (
            self.file_handle,
            self.datasets_index,
            self._size_dataset,
            self.meta,
        ) = get_h5_dataset(dataset_path=h5_path, meta_path=meta_path)

        # Optionally use a subset
        if isinstance(self.use_subset, int) and self.use_subset > 0:
            if self.use_subset < len(self.datasets_index):
                self.datasets_index = self.datasets_index[: self.use_subset]
                self._size_dataset = min(self._size_dataset, self.use_subset)
                logger.info(f"Using subset of size {self.use_subset}.")
            else:
                logger.warning(
                    f"Requested subset size ({self.use_subset}) >= available "
                    f"({len(self.datasets_index)}). Using full dataset."
                )
        # print size of dataset
        # logger.info(f"Dataset size: {self._size_dataset} frames.")
        # print(f"Dataset size: {self._size_dataset} frames.")

        # # print all the info about the first entry in the dataset
        # if self._size_dataset > 0:
        #     traj_number = self.datasets_index[0]
        #     traj = get_traj_as_meshes(
        #         file_handle=self.file_handle, traj_number=traj_number, meta=self.meta
        #     )
        #     # logger.info(f"First trajectory info: {traj}")
        #     # print(f"First trajectory info: {traj}")
        #     # print the size of the array for each key in the trajectory
        #     for key, value in traj.items():
                
        #         logger.info(f"{key}: {value.shape}")
        #         print(f"{key}: {value.shape}")
                


    @property
    def size_dataset(self) -> int:
        """Returns the number of trajectories in the dataset."""
        return self._size_dataset

    def __getitem__(self, index: int) -> Union[Data, Tuple[Data, torch.Tensor]]:
        """Retrieve a graph representation of a frame from a trajectory.

        This method extracts a single frame from a trajectory based on the index provided.
        It first determines the trajectory and frame number using `get_traj_frame` method.
        Then, it retrieves the trajectory data as meshes and converts the specified frame
        into a graph representation.

        Parameters:
            index (int): The index of the item in the dataset.

        Returns:
            Union[Data, Tuple[Data, torch.Tensor]]: A graph representation of the specified frame in the trajectory,
            optionally along with selected indices if masking is applied.
        """

        # temporary hack to prevent index out of bounds
        # if index+1 >= self.size_dataset:
        #     index = self.size_dataset - 1
        #     # print warning
        #     print(f"Warning: Index {index} is out of bounds, using last index instead.")
        
        traj_index, frame = self.get_traj_frame(index=index)
        traj_number = self.datasets_index[traj_index]

        # Retrieve the trajectory data
        traj = get_traj_as_meshes(
            file_handle=self.file_handle, traj_number=traj_number, meta=self.meta
        )

        # Get the graph for the specified frame
        graph = get_frame_as_graph(
            traj=traj, frame=frame, meta=self.meta, frame_target=frame + 1
        )

        if self.use_previous_data:
            previous_graph = get_frame_as_graph(
                traj=traj, frame=frame - 1, meta=self.meta, frame_target=None
            )
            graph.previous_data = previous_graph.x

        graph = graph.to(self.device)

        graph = self._apply_preprocessing(graph)
        graph = self._apply_k_hop(graph, traj_index)
        graph = self._may_remove_edges_attr(graph)
        graph = self._add_random_edges(graph)
        selected_indices = self._get_masked_indexes(graph)

        del graph.previous_data
        graph.traj_index = traj_index

        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph

    def __del__(self):
        """Ensure that the H5 file is properly closed."""
        if hasattr(self, "file_handle") and self.file_handle is not None:
            self.file_handle.close()
