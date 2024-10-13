import unittest
import pytorch_lightning as pl
from unittest.mock import MagicMock
import wandb
import numpy as np
from graphphysics.training.callback import LogPyVistaPredictionsCallback
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from tests.mock import (
    MOCK_XDMF_FOLDER,
    MOCK_H5_META10_SAVE_PATH,
)
from graphphysics.dataset.preprocessing import build_preprocessing


class MockModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, graph):
        return None, None, graph.x


transform = build_preprocessing(add_edges_features=True)
dataset = XDMFDataset(
    xdmf_folder=MOCK_XDMF_FOLDER,
    meta_path=MOCK_H5_META10_SAVE_PATH,
    preprocessing=transform,
)


class TestLogPyVistaPredictionsCallback(unittest.TestCase):
    def setUp(self):
        self.dataset = dataset
        self.indices = [0, 1]
        self.output_dir = "test_predictions"
        self.model = MockModel()
        self.callback = LogPyVistaPredictionsCallback(
            dataset=self.dataset, indices=self.indices, output_dir=self.output_dir
        )

        # Mock trainer and logger
        self.trainer = MagicMock()
        self.trainer.logger = MagicMock(spec=pl.loggers.WandbLogger)
        self.trainer.logger.experiment = MagicMock()
        self.trainer.logger.log = MagicMock()

    def test_on_validation_epoch_end(self):
        self.callback.on_validation_epoch_end(self.trainer, self.model)

        # Check that videos were logged
        self.assertTrue(self.trainer.logger.log.called)
        # Get the list of call arguments
        calls = self.trainer.logger.log.call_args_list
        # We expect two calls, one for predictions, one for ground truth
        self.assertEqual(len(calls), 4)

        # Check the first call
        args, kwargs = calls[2]
        log_dict = args[0] if args else kwargs
        self.assertIn("pyvista_predictions_video", log_dict)
        self.assertIsInstance(log_dict["pyvista_predictions_video"], wandb.Video)

        # Check the second call
        args, kwargs = calls[3]
        log_dict = args[0] if args else kwargs
        self.assertIn("pyvista_ground_truth_video", log_dict)
        self.assertIsInstance(log_dict["pyvista_ground_truth_video"], wandb.Video)

    def tearDown(self):
        import shutil
        import os

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)


if __name__ == "__main__":
    unittest.main()
