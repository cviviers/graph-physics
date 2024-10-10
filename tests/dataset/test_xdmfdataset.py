import unittest
import os

from graphphysics.dataset.xdmf_dataset import XDMFDataset
from tests.mock import (
    MOCK_XDMF_FOLDER,
    MOCK_H5_META10_SAVE_PATH,
)
from graphphysics.dataset.preprocessing import build_preprocessing


class TestH5Dataset(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
        )
        self.dataset.trajectory_length += 1

    def test_length(self):
        assert len(self.dataset) == 5

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index is None


class TestH5DatasetMasking(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            masking_ratio=0.4,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph, selected_index = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index is None
        assert selected_index is not None
        assert len(selected_index) == int((1 - 0.4) * 1923)


class TestH5DatasetPreprocessing(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            preprocessing=transform,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 11070)
        assert graph.edge_attr.shape == (11070, 4)
        assert graph.face is None


class TestH5DatasetPreprocessingNoEdgeFeatures(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=False)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            preprocessing=transform,
            add_edge_features=False,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 11070)
        assert graph.edge_attr is None
        assert graph.face is None


class TestH5DatasetPreprocessingKHOP(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            preprocessing=transform,
            khop=2,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 32638)
        assert graph.edge_attr.shape == (32638, 4)
        assert graph.face is None


class TestH5DatasetPreprocessingNoEdgeFeaturesKHOP(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=False)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            preprocessing=transform,
            khop=2,
            add_edge_features=False,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 32638)
        assert graph.edge_attr is None
        assert graph.face is None
